import torch
from PIL import Image
from torch import nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from open_clip import create_model_from_pretrained


class DFNCLIPWrapper(nn.Module):
    """
    DFN CLIP (Apple) backbone wrapper using OpenCLIP.
    """

    def __init__(self, name="DFN2B-CLIP-ViT-B-16", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        
        self.patch_size = 16 if "16" in name else 14
        hg_name = "hf-hub:apple/" + name
        
        # Load model using OpenCLIP
        self.model, _ = create_model_from_pretrained(hg_name)
        self.model.to(device)
        self.model.eval()
        
        # Get the image size from the model's image preprocessor config
        self.image_size = self.model.visual.image_size
        
        # Determine embed dimension based on model variant
        if "ViT-S" in name:
            self.embed_dim = 384
        elif "ViT-B" in name:
            self.embed_dim = 768
        elif "ViT-L" in name:
            self.embed_dim = 1024
        elif "ViT-H" in name:
            self.embed_dim = 1280
        else:
            raise ValueError(f"Unknown DFN CLIP model variant: {name}")

    def make_image_transform(self, img_size):
        """Create transform for DFN CLIP that matches OpenCLIP's preprocessing."""
        # OpenCLIP normalizes with ImageNet stats
        return T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
    def preprocess(self, img):
        """Preprocess image tensor for OpenCLIP model."""
        # Input is already a tensor from dataloader with normalization applied
        if img.dim() == 3:  # [C, H, W]
            img = img.unsqueeze(0)  # Add batch dimension
        return img.to(self.device)

    @torch.no_grad()
    def forward(self, img):
        x = self.preprocess(img)
        
        # Use OpenCLIP's encode_image but access intermediate features
        visual = self.model.visual
        
        # OpenCLIP's standard image encoding pipeline
        x = visual.conv1(x)  # Patch embedding: [B, embed_dim, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]
        
        # Prepend class token
        x = torch.cat([
            visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = x + visual.positional_embedding.to(x.dtype)
        
        # Pass through transformer
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)
        
        # Split CLS token and patch tokens
        cls = x[:, 0, :]  # [B, embed_dim]
        patch_tokens = x[:, 1:, :]  # [B, num_patches, embed_dim]
        
        # Reshape patch tokens to spatial format
        B, num_patches, C = patch_tokens.shape
        H = W = int(num_patches ** 0.5)
        assert H * W == num_patches, f"Mismatch: {H}*{W}={H*W} != {num_patches}"
        
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        return spatial, cls