import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training import get_batch, get_dataloaders
from tz_train_probes import UpsamplerWrapper


# ----------------------------
# Inference
# ----------------------------
class InferenceEngine:
    def __init__(self, backbone, classifier, upsampler, device, cfg):
        self.backbone = backbone
        self.classifier = classifier
        self.upsampler = upsampler
        self.device = device
        self.cfg = cfg

        # Freeze all models
        self.backbone.eval()
        self.classifier.eval()
        self.upsampler.eval()
        
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def predict_batch(self, batch):
        batch = get_batch(batch, self.device)
        image = batch["image"]
        
        # Forward pass through frozen backbone
        lr_feats, _ = self.backbone(image)
        if self.upsampler:
            features = self.upsampler(image, lr_feats)
        else:
            features = lr_feats

        # Get predictions from classifier
        pred = self.classifier(features)
        if pred.shape[-2:] != image.shape[-2:]:
            pred = F.interpolate(pred, size=image.shape[-2:], mode="bilinear")

        # Return logits and predicted labels
        pred_labels = pred.argmax(dim=1)
        return pred_labels, pred

    @torch.inference_mode()
    def predict_dataset(self, test_loader, output_dir, save_logits=False):
        """Run inference on test dataset and save predictions to disk incrementally"""
        os.makedirs(output_dir, exist_ok=True)
        
        pred_dir = os.path.join(output_dir, "pred_batches")
        os.makedirs(pred_dir, exist_ok=True)
        
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inferencing")):
            pred_labels, pred_logits = self.predict_batch(batch)
            
            # Save batch predictions immediately to disk (use uint8 to save space)
            batch_pred = pred_labels.cpu().numpy().astype(np.uint8)
            batch_path = os.path.join(pred_dir, f"batch_{batch_idx:05d}.npy")
            np.save(batch_path, batch_pred)
            
            total_samples += batch_pred.shape[0]
            torch.cuda.empty_cache()

        # Combine all batch files into single file
        print(f"\nCombining batch predictions into single file...")
        batch_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
        
        pred_path = os.path.join(output_dir, "predictions.npy")
        
        # Use memory-mapped array to avoid loading everything at once
        first_batch = np.load(os.path.join(pred_dir, batch_files[0]))
        mmap_predictions = np.lib.format.open_memmap(
            pred_path, mode='w+', dtype=first_batch.dtype, shape=(total_samples,) + first_batch.shape[1:]
        )
        
        idx = 0
        for batch_file in tqdm(batch_files, desc="Combining"):
            batch_pred = np.load(os.path.join(pred_dir, batch_file))
            mmap_predictions[idx:idx+len(batch_pred)] = batch_pred
            idx += len(batch_pred)
        
        mmap_predictions.flush()
        del mmap_predictions
        
        print(f"\nPredictions saved to {pred_path}")
        print(f"Total samples: {total_samples}")
        
        # Clean up batch files
        import shutil
        shutil.rmtree(pred_dir)
        
        return pred_path, total_samples


# ----------------------------
# Main
# ----------------------------
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load backbone
    print(f"Loading backbone: {cfg.backbone.name}")
    backbone = instantiate(cfg.backbone).to(device)
    backbone.eval()
    
    # Load test dataset
    print(f"Loading test dataset...")
    _, test_loader = get_dataloaders(cfg, backbone, is_evaluation=True)
    
    # Initialize upsampler
    upsampler = UpsamplerWrapper(cfg).to(device)
    
    # Load classifier weights
    if getattr(cfg.backbone, "adaptor_names", None):
        cls_filename = f"{cfg.backbone.name}-{cfg.backbone.adaptor_names}-{cfg.eval.upsampler}-{cfg.eval.task}-classifier.pth"
    else:
        cls_filename = f"{cfg.backbone.name}-{cfg.eval.upsampler}-{cfg.eval.task}-classifier.pth"
    
    checkpoint_path = os.path.join(
        cfg.project_root,
        "checkpoints/semseg-linear",
        cls_filename
    )
    
    print(f"Loading classifier from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}")
    
    classifier = nn.Conv2d(
        backbone.embed_dim,
        cfg.metrics.seg.num_classes if cfg.eval.task == "seg" else 1,
        1
    ).to(device)
    classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Run inference
    engine = InferenceEngine(backbone, classifier, upsampler, device, cfg)
    
    # Create output directory
    output_dir = os.path.join(cfg.project_root, "inference_outputs")
    print(f"Running inference on test set...")
    
    pred_path, total_samples = engine.predict_dataset(test_loader, output_dir, save_logits=False)
    
    print(f"Inference completed successfully!")
    print(f"Predictions saved at: {pred_path}")
    print(f"Total samples: {total_samples}")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import hydra
    @hydra.main(config_path="../config", config_name="eval", version_base=None)
    def run(cfg):
        main(cfg)
    run()