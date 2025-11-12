import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, JaccardIndex
from einops import rearrange
from hydra.utils import instantiate
from tqdm import tqdm
import numpy as np
import wandb
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training import get_batch, get_dataloaders

LOG_INTERVAL = 100

# ----------------------------
# Metrics for depth
# ----------------------------
def eval_metrics(gt, pred, min_depth=1e-3, max_depth=10):
    mask = (gt > min_depth) & (gt < max_depth)
    gt, pred = gt[mask], pred[mask]

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    silog = np.sqrt(np.mean((np.log(pred) - np.log(gt))**2) - np.mean(np.log(pred) - np.log(gt))**2) * 100
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, sq_rel=sq_rel,
                rmse=rmse, rmse_log=rmse_log, silog=silog, log_10=log_10)


# ----------------------------
# Upsampler Wrapper
# ----------------------------
class UpsamplerWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.method = cfg.eval.upsampler  # "none", "anyup", or "bilinear"
        if self.method == "anyup":
            self.anyup = torch.hub.load('wimmerth/anyup', 'anyup')
        # bilinear needs no extra module

    def forward(self, hr_image, lr_features):
        if self.method=="none":
            return lr_features
        elif self.method == "anyup":
            return self.anyup(hr_image, lr_features)
        elif self.method == "bilinear":
            return F.interpolate(lr_features, size=hr_image.shape[-2:], mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unknown upsampler: {self.method}")


# ----------------------------
# Evaluator
# ----------------------------
class UpsamplerEvaluator:
    def __init__(self, backbone, upsampler, device, cfg):
        self.backbone, self.upsampler, self.device, self.cfg = backbone, upsampler, device, cfg
        self.upsampler.to(device)
        self.backbone.to(device)

        # Freeze backbone and upsampler
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable classifier
        self.classifier = nn.Conv2d(self.backbone.embed_dim, # cfg.model.feature_dim,
                                    cfg.metrics.seg.num_classes if cfg.eval.task=="seg" else 1,
                                    1).to(device)
        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs)
        self.iou_metric = JaccardIndex(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
        self.accuracy_metric = Accuracy(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)

    def process_batch(self, batch):
        batch = get_batch(batch, self.device)
        image, target = batch["image"], batch["label"].to(self.device)

        # Forward pass through frozen backbone + model
        with torch.no_grad():
            lr_feats, _ = self.backbone(image)
            if self.upsampler:
                features = self.upsampler(image, lr_feats)
            else:
                features = lr_feats

        pred = self.classifier(features)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear")

        valid_mask = target != 255
        
        if self.cfg.eval.task == "seg":
            pred = rearrange(pred, "b c h w -> (b h w) c")
            target = rearrange(target, "b h w -> (b h w)")
            valid_mask = rearrange(valid_mask, "b h w -> (b h w)")
        pred = pred[valid_mask]
        target = target[valid_mask]
        return pred, target

    def train(self, train_loader, epoch):
        self.classifier.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            pred, target = self.process_batch(batch)
            if self.cfg.eval.task == "seg":
                loss = F.cross_entropy(pred, target)
            else:
                loss = F.mse_loss(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"[Epoch {epoch}] Batch {batch_idx+1} | Train Loss: {avg_loss:.6f}")
                lr = self.optimizer.param_groups[0]["lr"]
                wandb.log({"train_loss": avg_loss, "batch": batch_idx + 1, "lr":lr})
        self.scheduler.step()
        print(f"[Epoch {epoch}] Training finished | Avg Loss: {total_loss/len(train_loader):.6f}")

    @torch.inference_mode()
    def evaluate(self, val_loader, epoch):
        self.classifier.eval()
        results = {}
        nsamples = 0
        if self.cfg.eval.task == "seg":
            self.iou_metric.reset()
            self.accuracy_metric.reset()
        for batch in tqdm(val_loader, desc="Evaluating",mininterval=5):
            pred, target = self.process_batch(batch)
            if self.cfg.eval.task == "seg":
                pred_labels = pred.argmax(dim=1)
                self.iou_metric.update(pred, target)
                self.accuracy_metric.update(pred_labels, target)
            else:
                gt = target.cpu().numpy().reshape(-1)
                pd = pred.cpu().numpy().reshape(-1)
                cur = eval_metrics(gt, pd)
                for k, v in cur.items():
                    results[k] = results.get(k, 0) + v
            nsamples += 1
        if self.cfg.eval.task == "seg":
            results["val_mIoU"]=self.iou_metric.compute().item()
            results["val_accuracy"]=self.accuracy_metric.compute().item()
        # Average over batches
        if self.cfg.eval.task != "seg":
            for k in results:
                results[k] /= nsamples

        print(f"[Epoch {epoch}] Evaluation Results: {results}")
        wandb.log(results)


# ----------------------------
# Main
# ----------------------------
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = instantiate(cfg.backbone).to(device)
    backbone.eval()

    train_loader, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True)

    upsampler = UpsamplerWrapper(cfg).to(device)
    evaluator = UpsamplerEvaluator(backbone, upsampler, device, cfg)
    if getattr(cfg.backbone, "adaptor_names", None):
        wandb_name=f"bb:{cfg.backbone.name}-adp:{cfg.backbone.adaptor_names}-res:{cfg.img_size}"
    else:
        wandb_name=f"bb:{cfg.backbone.name}-res:{cfg.img_size}"
    wandb.init(project=f"{cfg.dataset_evaluation.tag}-{cfg.eval.task}-lp",
               name=wandb_name, config=OmegaConf.to_container(cfg, resolve=False))

    for epoch in range(cfg.num_epochs):
        evaluator.train(train_loader, epoch)
        evaluator.evaluate(val_loader, epoch)

    # Save trained classifier
    try:
        checkpoints_folder="checkpoints/semseg-linear"
        if getattr(cfg.backbone, "adaptor_names", None):
            cls_filename=f"{cfg.backbone.name}-{cfg.backbone.adaptor_names}-{cfg.eval.upsampler}-{cfg.eval.task}-classifier.pth"
        else: 
            cls_filename=f"{cfg.backbone.name}-{cfg.eval.upsampler}-{cfg.eval.task}-classifier.pth"
        save_path = os.path.join(
            cfg.project_root, 
            checkpoints_folder, 
            cls_filename
        )
        torch.save(evaluator.classifier.state_dict(), save_path)
        print(f"Training completed and classifier saved at {save_path}")
    except Exception as e:
        print(f"WARNING: Saving failed: {e}")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    import hydra
    @hydra.main(config_path="../config", config_name="eval")
    def run(cfg):
        main(cfg)
    run()
