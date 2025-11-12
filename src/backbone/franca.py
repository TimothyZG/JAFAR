import torch
from PIL import Image
from torch import nn
import torchvision.transforms as T

class FrancaWrapper(nn.Module):
    """
    Franca backbone wrapper.
    """

    def __init__(self, name="franca_vitb14", device="cuda", weights="LAION", use_rasa_head=True):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=14 # All Franca ViT has patch size 14
        self.model = torch.hub.load('valeoai/Franca', name, weights=weights, use_rasa_head=use_rasa_head)
        self.model.to(device)
        self.model.eval()
        self.weights=weights
        self.use_rasa_head = use_rasa_head
        embed_map = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536}
        for k,v in embed_map.items():
            if k in name:
                self.embed_dim = v
                break
        else:
            raise ValueError(f"Unknown Franca model variant: {name}")

    def make_image_transform(self, img_size):
        return T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.no_grad()
    def forward(self, img):
        img = img.to(self.device)
        feats = self.model.forward_features(img,use_rasa_head=self.use_rasa_head)
        cls_token = feats["x_norm_clstoken"]
        patch_tokens = feats["x_norm_patchtokens"]
        patch_tokens_debiased = feats["patch_token_rasa"]
        
        B, num_patches, C = patch_tokens.shape
        _,_,processed_height,processed_width = img.shape
        H = processed_height // self.patch_size
        W = processed_width // self.patch_size
        assert H * W == num_patches, f"Mismatch: {H}*{W}={H*W} != {num_patches}"
        
        spatial = patch_tokens_debiased.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial, cls_token