import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class RadioWrapper(nn.Module):
    """
    RADIO backbone wrapper.
    Note: Input must be a PIL Image (tz edit: tensor, not image). RADIO expects input values in [0, 1] (float32).
    RADIO will automatically normalize to mean 0, std 1 internally.
    """

    def __init__(self, name="radio_v2.5-b", device="cuda", adaptor_name=None, res=512):
        super().__init__()
        self.name = name
        self.device = device
        self.adaptor_name=adaptor_name if adaptor_name!="backbone" else None
        self.res=res
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=name,
            progress=True,
            skip_validation=True,
            adaptor_names=adaptor_name if adaptor_name!="backbone" else None
        )
        self.config = {"mean": torch.tensor([0.0, 0, 0]), "std": torch.tensor([1.0, 1, 1])}  # RADIO normalizes internally
        self.model.to(self.device).eval()
        if name == "radio_v2.5-h":
            self.embed_dim = 1280
        elif name == "radio_v2.5-b":
            if adaptor_name=="sam":
                self.embed_dim = 1280
            elif adaptor_name=="dino_v2":
                self.embed_dim = 1536
            elif adaptor_name=="siglip":
                self.embed_dim = 1152
            elif adaptor_name=="clip":
                self.embed_dim = 1280
            else: self.embed_dim = 768
        self.patch_size = 1
    def get_identifiable_name(self):
        adaptor_name=self.adaptor_name if self.adaptor_name else "backbone"
        return self.name+f"-{adaptor_name}"
    
    def make_image_transform(self):
        """Create transform for RADIO - resize for batching, convert to tensor."""
        return T.Compose([
            T.Resize(self.res, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop((self.res, self.res)),
            T.ToTensor()
        ])
    def preprocess(self, img: Image.Image):
        x = img
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)
        if "e-radio" in self.name:
            self.model.model.set_optimal_window_size(x.shape[2:])
        return x

    @torch.no_grad()
    def forward(self, img):
        x = self.preprocess(img)
        # Only return spatial_features in NCHW format
        out = self.model(x, feature_fmt="NCHW")
        if self.adaptor_name:
            _, spatial_features = out[self.adaptor_name]
        else:
            _, spatial_features = out
        assert spatial_features.ndim == 4
        return spatial_features, None
