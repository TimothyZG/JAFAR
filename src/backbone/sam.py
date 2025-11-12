import torch
from PIL import Image
from torch import nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, SiglipVisionModel, Siglip2VisionModel
import timm


class SamWrapper(nn.Module):
    """
    SigLIP backbone wrapper.
    """

    def __init__(self, name="samvit_base_patch16.sa1b", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=16 # All SAM ViTs has patch size 16
        self.model=timm.create_model(name,pretrained=True,num_classes=0)
        self.model.to(device)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.model.eval()
        if "base" in name:
            self.embed_dim = 256
        elif "large" in name:
            self.embed_dim = 512
        elif "huge" in name:
            self.embed_dim = 1024
        else:
            raise ValueError(f"Unknown SAM model variant: {name}")

    def make_image_transform(self, img_size):
        """Create transform for SigLIP - resize for batching, convert to tensor."""
        return self.transforms

    @torch.no_grad()
    def forward(self, img):
        spacial = self.model.forward_features(img) # B,C,H,W
        cls = self.model.forward_head(spacial, pre_logits=True)
        B,C,H,W = spacial.shape
        expected = img.shape[2] // self.patch_size         # e.g., 1024/16 = 64
        assert H == expected and W == expected, f"Expected {expected}x{expected}, got {H}x{W}"
        return spacial, cls