import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoImageProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class DinoV2Wrapper(nn.Module):
    """
    DinoV2 backbone wrapper.
    """

    def __init__(self, name="dinov2-small", device="cuda", res=518):
        super().__init__()
        self.name = name
        self.device = device
        self.patch_size=14 # all dino v2 models have patch size=14
        self.res = res
        self.model = AutoModel.from_pretrained(f"facebook/{name}")
        self.model.to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(f"facebook/{name}", do_rescale=False, do_center_crop=False, do_resize=False)
        self.config = {"mean": torch.tensor([0.0, 0, 0]), "std": torch.tensor([1.0, 1, 1])}
        
        if "small" in name:
            self.embed_dim = 384
        elif "base" in name:
            self.embed_dim = 768
        elif "large" in name:
            self.embed_dim = 1024
        elif "giant" in name:
            self.embed_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv2 model variant: {name}")
    def get_identifiable_name(self):
        return self.name
    
    def make_image_transform(self):
        """Create transform for RADIO - resize for batching, convert to tensor."""
        return T.Compose([
            T.Resize(self.res, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop((self.res, self.res)),
            T.ToTensor(),
        ])
        
    def preprocess(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)
        return inputs

    @torch.no_grad()
    def forward(self, img: Image.Image):
        img=self.preprocess(img)
        batch_size, rgb, img_height, img_width = img.pixel_values.shape
        num_patches_height, num_patches_width = img_height // self.patch_size, img_width // self.patch_size
        num_patches_flat = num_patches_height * num_patches_width
        outputs = self.model(**img)
        last_hidden_states = outputs.last_hidden_state
        assert last_hidden_states.shape == (batch_size, 1 + num_patches_flat, self.embed_dim)
        
        cls_token = last_hidden_states[:, 0, :]
        patch_features = last_hidden_states[:, 1:, :].permute(0, 2, 1).reshape(batch_size, self.embed_dim, num_patches_height, num_patches_width)
        
        return patch_features, cls_token