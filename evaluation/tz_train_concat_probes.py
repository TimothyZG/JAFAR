import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, JaccardIndex
from einops import rearrange
from hydra.utils import instantiate
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from itertools import zip_longest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training import get_batch, get_dataloaders

LOG_INTERVAL = 100


# ----------------------------
# Feature Concatenator Module
# ----------------------------
class FeatureConcatenator(nn.Module):
    """Concatenates features from multiple backbones after upsampling to max spatial size."""

    def __init__(self, backbones, device):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.device = device
        self.num_backbones = len(backbones)

        # Calculate total concatenated dimension
        self.concat_dim = sum(bb.embed_dim for bb in backbones)

        # Freeze all backbones
        for backbone in self.backbones:
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, images_list):
        """
        Args:
            images_list: List of images, one per backbone (each with backbone-specific transforms)

        Returns:
            Concatenated features upsampled to max spatial size
        """
        features_list = []

        # Extract features from each backbone
        for backbone, image in zip(self.backbones, images_list):
            lr_feats, _ = backbone(image)
            features_list.append(lr_feats)

        # Find max spatial size
        max_h = max(f.shape[2] for f in features_list)
        max_w = max(f.shape[3] for f in features_list)

        # Upsample all features to max size and concatenate
        upsampled_features = []
        for feats in features_list:
            if feats.shape[2:] != (max_h, max_w):
                feats = F.interpolate(feats, size=(max_h, max_w), mode='bilinear', align_corners=False)
            upsampled_features.append(feats)

        # Concatenate along channel dimension
        concat_feats = torch.cat(upsampled_features, dim=1)  # (B, sum(embed_dims), H, W)

        return concat_feats


# ----------------------------
# Concatenated Probe Trainer
# ----------------------------
class ConcatProbeTrainer:
    def __init__(self, feature_concatenator, device, cfg):
        self.feature_concatenator = feature_concatenator.to(device)
        self.device = device
        self.cfg = cfg

        # Trainable classifier: concatenated_dim -> num_classes
        if cfg.head=="lp":
            self.classifier = nn.Conv2d(
                feature_concatenator.concat_dim,
                cfg.metrics.seg.num_classes,
                1
            ).to(device)
        elif cfg.head=="mlp":
            self.classifier = nn.Sequential(
            nn.Conv2d(
                feature_concatenator.concat_dim,
                feature_concatenator.concat_dim,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_concatenator.concat_dim),
            nn.GELU(),
            nn.Conv2d(
                feature_concatenator.concat_dim,
                cfg.metrics.seg.num_classes,
                1
            )
            ).to(device)
        else:
            raise NotImplementedError(f"Unrecognized head type {cfg.head=}")

        # Print parameter count
        total_params = sum(p.numel() for p in self.classifier.parameters())
        trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f"\nClassifier head '{cfg.head}' parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Input dim: {feature_concatenator.concat_dim}")
        print(f"  Output classes: {cfg.metrics.seg.num_classes}\n")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs
        )

        # Metrics
        self.iou_metric = JaccardIndex(
            num_classes=cfg.metrics.seg.num_classes,
            task="multiclass"
        ).to(device)
        self.accuracy_metric = Accuracy(
            num_classes=cfg.metrics.seg.num_classes,
            task="multiclass"
        ).to(device)

    def process_batch(self, batches_list):
        """
        Process a list of batches (one per backbone) and return predictions and targets.

        Args:
            batches_list: List of batches, one per backbone

        Returns:
            pred: Predictions (flattened and masked)
            target: Targets (flattened and masked)
        """
        # Move all batches to device and extract images
        images_list = []
        target = None

        for batch in batches_list:
            batch = get_batch(batch, self.device)
            images_list.append(batch["image"])
            if target is None:
                target = batch["label"].to(self.device)

        # Get concatenated features from all backbones
        with torch.no_grad():
            concat_feats = self.feature_concatenator(images_list)

        # Pass through trainable classifier
        pred = self.classifier(concat_feats)

        # Upsample prediction to target size if needed
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

        # Flatten and mask
        valid_mask = target != 255
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b h w -> (b h w)")
        valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

        pred = pred[valid_mask]
        target = target[valid_mask]

        return pred, target

    def train(self, dataloaders, epoch):
        """Train for one epoch using multiple dataloaders."""
        self.classifier.train()
        total_loss = 0

        # Create iterators for all dataloaders
        dataloader_iters = [iter(dl) for dl in dataloaders]
        num_batches = len(dataloaders[0])

        batch_idx = 0
        for batches_list in zip_longest(*dataloader_iters):
            # Check if any dataloader is exhausted
            if any(b is None for b in batches_list):
                break

            pred, target = self.process_batch(batches_list)
            loss = F.cross_entropy(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"[Epoch {epoch}] Batch {batch_idx+1}/{num_batches} | Train Loss: {avg_loss:.6f}")
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log({"train_loss": avg_loss, "batch": batch_idx + 1, "lr": lr})

            batch_idx += 1

        self.scheduler.step()
        avg_loss = total_loss / batch_idx if batch_idx > 0 else 0
        print(f"[Epoch {epoch}] Training finished | Avg Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.inference_mode()
    def evaluate(self, dataloaders, epoch):
        """Evaluate using multiple dataloaders."""
        self.classifier.eval()
        self.iou_metric.reset()
        self.accuracy_metric.reset()

        # Create iterators for all dataloaders
        dataloader_iters = [iter(dl) for dl in dataloaders]
        num_batches = len(dataloaders[0])

        for batches_list in tqdm(zip_longest(*dataloader_iters), total=num_batches, desc="Evaluating", mininterval=5):
            # Check if any dataloader is exhausted
            if any(b is None for b in batches_list):
                break

            pred, target = self.process_batch(batches_list)
            pred_labels = pred.argmax(dim=1)

            self.iou_metric.update(pred, target)
            self.accuracy_metric.update(pred_labels, target)

        results = {
            "val_mIoU": self.iou_metric.compute().item(),
            "val_accuracy": self.accuracy_metric.compute().item(),
            "epoch": epoch
        }

        print(f"[Epoch {epoch}] Evaluation Results: {results}")
        wandb.log(results)

        return results


# ----------------------------
# Main
# ----------------------------
def main(cfg):
    print("\n" + "="*100)
    print("CONCATENATED PROBE TRAINING CONFIG")
    print("="*100)
    print(OmegaConf.to_yaml(cfg))
    print("="*100 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all backbones from config
    backbone_names = cfg.concat_ensemble.backbones
    print(f"Loading {len(backbone_names)} backbones: {backbone_names}")

    backbones = []
    train_dataloaders = []
    val_dataloaders = []

    for bb_name in backbone_names:
        # Load backbone config
        backbone_cfg = OmegaConf.load(
            os.path.join(cfg.concat_ensemble.backbone_dir, f"{bb_name}.yaml")
        )
        backbone = instantiate(backbone_cfg)
        backbones.append(backbone)

        train_loader, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)

        print(f"  Loaded backbone: {bb_name} (embed_dim={backbone.embed_dim})")

    print(f"\nLoaded {len(train_dataloaders)} train dataloaders")
    print(f"Loaded {len(val_dataloaders)} val dataloaders\n")

    # Create feature concatenator
    feature_concatenator = FeatureConcatenator(backbones, device)
    print(f"Total concatenated feature dimension: {feature_concatenator.concat_dim}")

    # Create trainer
    trainer = ConcatProbeTrainer(feature_concatenator, device, cfg)

    # Initialize wandb
    backbone_str = "+".join(backbone_names)
    wandb_name = f"concat_{cfg.head}:{backbone_str}-res:{cfg.target_size}"
    wandb.init(
        project=f"{cfg.dataset_evaluation.tag}-seg-concat-lp",
        name=wandb_name,
        config=OmegaConf.to_container(cfg, resolve=False)
    )

    # Training loop
    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    print("="*100)

    best_miou = 0.0
    for epoch in range(cfg.num_epochs):
        print(f"\n{'='*100}")
        print(f"EPOCH {epoch + 1}/{cfg.num_epochs}")
        print(f"{'='*100}")

        trainer.train(train_dataloaders, epoch)
        results = trainer.evaluate(val_dataloaders, epoch)

        # Track best model
        if results["val_mIoU"] > best_miou:
            best_miou = results["val_mIoU"]
            print(f"New best mIoU: {best_miou:.4f}")

    # Save trained classifier
    try:
        checkpoints_folder = "checkpoints/concat-linear-probes"
        os.makedirs(os.path.join(cfg.project_root, checkpoints_folder), exist_ok=True)

        backbone_str_filename = "_".join(backbone_names)
        cls_filename = f"concat_{backbone_str_filename}_seg_classifier.pth"
        save_path = os.path.join(cfg.project_root, checkpoints_folder, cls_filename)

        torch.save(trainer.classifier.state_dict(), save_path)
        print(f"\n{'='*100}")
        print(f"Training completed!")
        print(f"Best mIoU: {best_miou:.4f}")
        print(f"Classifier saved at: {save_path}")
        print(f"{'='*100}\n")

        wandb.log({"best_val_mIoU": best_miou})

    except Exception as e:
        print(f"WARNING: Saving failed: {e}")

    wandb.finish()


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="../config", config_name="concat_lp", version_base=None)
    def run(cfg):
        main(cfg)

    run()
