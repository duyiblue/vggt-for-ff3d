import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from .aggregator import Aggregator
from ..heads.dpt_head import DPTHead
from .vggt import VGGT

class VGGT_For_FF3D(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, 
                 semantic_label_dim=512, nocs_num_bins=64,
                 k=2, sh_degree=0):
        """
        Args:
            img_size (int): This is only to set a default size. In runtime, the model supports varying (and non-square) sizes.
            patch_size (int): ViT patch size. 14 is the default for DINOv2.
            embed_dim (int): Dimension of ViT tokens, used throughout the transformer (Aggregator) part.
            semantic_label_dim (int): Dimension of the semantic label embeddings (512 for CLIP).
            nocs_num_bins (int): Number of bins for NOCS classification (e.g., 64).
            k (int): Number of Gaussians per pixel (e.g., 2).
            sh_degree (int): Degree of the spherical harmonics for each Gaussian, related to color.
        """

        super().__init__()

        # Keep the aggregator the same as VGGT's
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # output_dim=2 because there's an additional channel for confidence.
        # For now we simply ignore the confidence output. In the future we can use it or modify the architecture to make confidence prediction optional.
        # For future modifications, work in layers/dpt_head.py and heads/head_act.py.
        self.depth_head = DPTHead(dim_in = 2 * embed_dim, 
                                  output_dim = 1 + 1,   # 1 channel for depth, 1 channel for confidence
                                  activation = "exp", 
                                  conf_activation = "expp1")

        # Treat semantic label prediction as a regression task.
        self.semantic_head = DPTHead(dim_in = 2 * embed_dim, 
                                     output_dim = semantic_label_dim + 1, 
                                     activation = "linear", 
                                     conf_activation = "expp1",
                                     features = 512,  # To predict the long CLIP label, we need more channels
                                     out_channels = [512, 1024, 2048, 2048])
        
        # Treat NOCS prediction as a hybrid classification + regression task.
        # First classify the bin, then refine the offset within the bin.
        self.nocs_classification_head = DPTHead(dim_in = 2 * embed_dim,
                                                output_dim = nocs_num_bins + 1,
                                                activation = "linear",  # Use linear for logits, will apply cross entropy loss in the training script
                                                conf_activation = "expp1")
        # Offset head for continuous refinement within bins
        self.nocs_regression_head = DPTHead(dim_in = 2 * embed_dim,
                                            output_dim = 3 + 1,
                                            activation = "linear",  # Unbounded offset. This is the simplest approach. Could try other activations in the future.
                                            conf_activation = "expp1")


        self.gaussian_basic_head = DPTHead(dim_in = 2 * embed_dim,
                                           output_dim = k * 11 + 1,  # 11 channels for mean(3) + scale(3) + rotation(4) + opacity(1)
                                           activation = "linear",
                                           conf_activation = "expp1")
        
        self.gaussian_sh_head = DPTHead(dim_in = 2 * embed_dim,
                                        output_dim = k * 3 * ((sh_degree + 1) ** 2) + 1,  # k Gaussians, each have 3 (RGB) channels
                                        activation = "linear",
                                        conf_activation = "expp1")

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [B, 3, H, W], in range [0, 1].
                B: batch size, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, 1, H, W]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, 3, H, W]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """
        if len(images.shape) != 4 or images.shape[1] != 3:
            raise ValueError(f"Input images must have shape [B, 3, H, W], but got {images.shape}")
        
        # VGGT processes multiple images in each sample, so we add a sequence length dimension for compatibility.
        # In the future, we'll modify the architecture to make this more elegant.
        images = images.unsqueeze(1)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            # VGGT disables autocast here, so we follow this.
            # This is the safest approach, but could be inefficient.
            # Could be modified in the future.

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth  # [B, 1, 1, H, W]
                predictions["depth_conf"] = depth_conf  # [B, 1, H, W]

            if self.semantic_head is not None:
                semantic, semantic_conf = self.semantic_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["semantic"] = semantic  # [B, 1, semantic_label_dim, H, W]
                predictions["semantic_conf"] = semantic_conf  # [B, 1, H, W]

            if self.nocs_classification_head is not None:
                nocs_bins, nocs_bins_conf = self.nocs_classification_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["nocs_bins"] = nocs_bins  # [B, 1, nocs_num_bins, H, W]
                predictions["nocs_bins_conf"] = nocs_bins_conf  # [B, 1, H, W]
            
            if self.nocs_regression_head is not None:
                nocs_offsets, nocs_offsets_conf = self.nocs_regression_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["nocs_offsets"] = nocs_offsets  # [B, 1, 3, H, W]
                predictions["nocs_offsets_conf"] = nocs_offsets_conf  # [B, 1, H, W]
            
            if self.gaussian_basic_head is not None:
                gaussian_basic, gaussian_basic_conf = self.gaussian_basic_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["gaussian_basic"] = gaussian_basic  # [B, 1, k * 11, H, W]
                predictions["gaussian_basic_conf"] = gaussian_basic_conf  # [B, 1, H, W]
            
            if self.gaussian_sh_head is not None:
                gaussian_sh, gaussian_sh_conf = self.gaussian_sh_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["gaussian_sh"] = gaussian_sh  # [B, 1, k * 3 * ((sh_degree + 1) ** 2), H, W]
                predictions["gaussian_sh_conf"] = gaussian_sh_conf  # [B, 1, H, W]
        
        # Remove the sequence dimension (S=1) that we added for VGGT compatibility
        # Convert [B, 1, ...] → [B, ...] for all predictions
        for key, value in predictions.items():
            if value.dim() < 2 or value.shape[1] != 1:
                raise ValueError("Something went wrong.")
            predictions[key] = value.squeeze(1)  # Remove dimension 1 (sequence dimension)

        # VGGT does this, but it depends on how we write the training script.
        # For now we eliminate this for simplicity.
        # if not self.training:
        #     predictions["images"] = images  # store the images for visualization during inference
                
        return predictions

def initialize_with_pretrained(model: VGGT_For_FF3D):
    """
    Initialize the model with pretrained VGGT weights.
    Modify the model in-place, return nothing.
    """
    pretrained_model = VGGT.from_pretrained("facebook/VGGT-1B")
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def create_model(load_pretrained: bool = False,
                 device: torch.device = torch.device("cpu")):
    model = VGGT_For_FF3D()
    if load_pretrained:
        initialize_with_pretrained(model)
    
    model = model.to(device)
    model.eval()

    return model

if __name__ == "__main__":
    model = create_model(load_pretrained=False)
    dummy_images = torch.randn(2, 3, 518, 518)
    predictions = model(dummy_images)
    for key, value in predictions.items():
        print(key, value.shape)
