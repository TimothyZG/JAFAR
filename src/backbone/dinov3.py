import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoImageProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class DinoV3Wrapper(nn.Module):
    """
    DinoV3 backbone wrapper.
    """

    def __init__(self, name="dinov3-vits16-pretrain-lvd1689m", device="cuda"):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=16 # All DinoV3 Vit has patch size 16
        hg_name="facebook/"+name
        self.model = AutoModel.from_pretrained(hg_name)
        self.model.to(device)
        self.processor = AutoImageProcessor.from_pretrained(hg_name, do_rescale=False, do_center_crop=False, do_resize=False)
        
        self.model.eval()
        if "vits" in name:
            self.embed_dim = 384
        elif "vitb" in name:
            self.embed_dim = 768
        elif "vitl" in name:
            self.embed_dim = 1024
        elif "vitg" in name:
            self.embed_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv3 model variant: {name}")

    def make_image_transform(self, img_size):
        """Create transform for RADIO - resize for batching, convert to tensor."""
        return T.Compose([
            T.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor()
        ])
        
    def preprocess(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)
        return inputs

    @torch.no_grad()
    def forward(self, img):
        x = self.preprocess(img)
        out = self.model(**x).last_hidden_state
        patch_tokens = out[:, 5:, :]
        
        B, num_patches, C = patch_tokens.shape
        processed_height = x['pixel_values'].shape[2]
        processed_width = x['pixel_values'].shape[3]
        H = processed_height // self.patch_size
        W = processed_width // self.patch_size
        assert H * W == num_patches, f"Mismatch: {H}*{W}={H*W} != {num_patches}"
        
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial, out[:, 0]