import torch
from PIL import Image
from torch import nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, SiglipVisionModel, Siglip2VisionModel


class SigLIPWrapper(nn.Module):
    """
    SigLIP backbone wrapper.
    """

    def __init__(self, name="siglip-base-patch16-512", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=16 # All siglip ViTs has patch size 16
        hg_name="google/"+name
        if "siglipv2" in name:
            self.model = Siglip2VisionModel.from_pretrained(hg_name)
        else:
            self.model = SiglipVisionModel.from_pretrained(hg_name)
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(hg_name, do_rescale=False, do_center_crop=False, do_resize=False)
        
        self.model.eval()
        if "small" in name:
            self.embed_dim = 384
        elif "base" in name:
            self.embed_dim = 768
        elif "large" in name:
            self.embed_dim = 1024
        elif "so400m" in name:
            self.embed_dim = 1152
        else:
            raise ValueError(f"Unknown SigLIP model variant: {name}")

    def make_image_transform(self, img_size):
        """Create transform for SigLIP - resize for batching, convert to tensor."""
        return T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor()
        ])
        
    def preprocess(self, img: Image.Image):
        inputs = self.processor(images=img, do_rescale=False, return_tensors="pt").to(self.model.device)
        return inputs

    @torch.no_grad()
    def forward(self, img):
        x = self.preprocess(img)
        outputs = self.model(**x)
        patch_tokens = outputs.last_hidden_state
        
        B, num_patches, C = patch_tokens.shape
        processed_height = x['pixel_values'].shape[2]
        processed_width = x['pixel_values'].shape[3]
        H = processed_height // self.patch_size
        W = processed_width // self.patch_size
        assert H * W == num_patches, f"Mismatch: {H}*{W}={H*W} != {num_patches}"
        
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        cls = outputs.pooler_output
        return spatial, cls