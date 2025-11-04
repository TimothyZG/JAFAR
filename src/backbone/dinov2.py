import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class DinoV2Wrapper(nn.Module):
    """
    DinoV2 backbone wrapper.
    """

    def __init__(self, name="dinov2_vits14", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=int(name[-2:])
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            self.name,
        )
        self.model.to(self.device).eval()
        if "vits14" in name:
            self.embed_dim = 384
        elif "vitb14" in name:
            self.embed_dim = 768
        elif "vitl14" in name:
            self.embed_dim = 1024
        elif "vitg14" in name:
            self.embed_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv2 model variant: {name}")
        
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def preprocess(self, img: Image.Image):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)
        h, w = img.shape[-2:]
        max_size = 518
        scale = min(max_size / max(h, w), 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        new_h = (new_h // self.patch_size) * self.patch_size
        new_w = (new_w // self.patch_size) * self.patch_size
        
        img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return self.normalize(img)

    @torch.no_grad()
    def forward(self, img: Image.Image):
        x = self.preprocess(img)
        feats = self.model.forward_features(x)
        cls_token = feats["x_norm_clstoken"]    # [B, C]
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, C]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        spatial_features = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial_features, cls_token