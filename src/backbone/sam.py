import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import timm


class SamWrapper(nn.Module):
    """
    SAM backbone wrapper.
    """

    def __init__(self, name="samvit_base_patch16.sa1b", device="cuda",res=1024):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=16 # All SAM ViTs has patch size 16
        self.model=timm.create_model(name,pretrained=True,num_classes=0)
        self.model.to(device)
        self.res=res
        self.model.eval()
        if "base" in name:
            self.embed_dim = 256 # 768 note we're using Sam's neck, which contains normalization.
        elif "large" in name:
            self.embed_dim = 1024
        else:
            raise ValueError(f"Unknown SAM model variant: {name}")

    def get_identifiable_name(self):
        return self.name
    
    def make_image_transform(self):
        """Create transform for SAM - just use timm's create trasform"""
        return T.Compose([
            T.Resize(self.res, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop((self.res, self.res)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711))
        ])

    @torch.no_grad()
    def forward(self, img):
        x = self.model.patch_embed(img)
        x = self.model.pos_drop(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = x.permute(0,3,1,2) # B,C,H,W is the desired output shape
        spacial = self.model.neck(x)
        cls = self.model.head(x)
        B,C,H,W = spacial.shape
        expected = img.shape[2] // self.patch_size         # e.g., 1024/16 = 64
        assert H == expected and W == expected, f"Expected {expected}x{expected}, got {H}x{W}"
        return spacial, cls