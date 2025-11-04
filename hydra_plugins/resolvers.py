from omegaconf import OmegaConf


def get_feature(target: str, adaptor_names: str) -> int:
    """Resolve feature dimensions from backbone name patterns"""
    model_name = target.lower()
    if " " in model_name and adaptor_names is None:
        model_name, adaptor_names = model_name.split(" ", 1)


    if "vits" in model_name or "vit_small" in model_name:
        return 384
    if "vitb" in model_name or "vit_base" in model_name:
        return 768
    if "vitl" in model_name or "vit_large" in model_name:
        return 1024
    if model_name == "efficientnet_b4":
        return 128
    if model_name == "maskclip":
        return 512
    if model_name == "radio_v2.5-h":
        return 1280
    if model_name == "radio_v2.5-l":
        return 1024
    if model_name == "radio_v2.5-b":
        if adaptor_names=="sam":
            return 1280
        elif adaptor_names=="dino_v2":
            return 1536
        elif adaptor_names=="siglip2":
            return 1152
        elif adaptor_names=="clip":
            return 1280
        return 768

    raise ValueError(f"Unsupported backbone: {model_name}")


OmegaConf.register_new_resolver("get_feature", lambda target, adaptor_names=None: get_feature(target, adaptor_names))


