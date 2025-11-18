import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, JaccardIndex
from einops import rearrange
from hydra.utils import instantiate
from tqdm import tqdm
import pandas as pd
from omegaconf import OmegaConf
from itertools import combinations, zip_longest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training import get_batch, get_dataloaders


class EnsembleEvaluator:
    def __init__(self, backbones, classifier_paths, device, cfg):
        """
        Args:
            backbone_configs: list of instantiated backbone objects
            classifier_paths: list of paths to trained classifier checkpoints
            device: torch device
            cfg: main config
        """
        self.device = device
        self.cfg = cfg
        self.eval_img_size=cfg.target_size # note this is different than individual model's input size;
        # We bilinearly upsample/downsample prediction to target_size for ensembling.
        self.num_models = len(backbones)
        
        # Load all backbones and classifiers
        self.backbones = []
        self.classifiers = []
        for backbone, cls_path in zip(backbones, classifier_paths):
            backbone.to(device)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            self.backbones.append(backbone)
            
            # Load classifier
            classifier = nn.Conv2d(backbone.embed_dim, cfg.dataset_evaluation.num_classes, 1).to(device)
            classifier.load_state_dict(torch.load(cls_path, map_location=device))
            classifier.eval()
            for p in classifier.parameters():
                p.requires_grad = False
            self.classifiers.append(classifier)
        
        # Initialize metrics for each ensemble configuration
        # Key: tuple of model indices (e.g., (0,), (1,), (0,1), etc.)
        self.ensemble_metrics = {}
        self.batch_predictions = {}  # Store logits for current batch across all models
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics for all possible ensemble combinations"""
        # Single models
        for i in range(self.num_models):
            key = (i,)
            self.ensemble_metrics[key] = {
                'iou_majvote': JaccardIndex(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                'acc_majvote': Accuracy(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                'iou_oracle': JaccardIndex(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                'acc_oracle': Accuracy(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
            }
        
        # All ensemble combinations (size 2 and above)
        for size in range(2, self.num_models + 1):
            for combo in combinations(range(self.num_models), size):
                self.ensemble_metrics[combo] = {
                    'iou_majvote': JaccardIndex(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                    'acc_majvote': Accuracy(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                    'iou_oracle': JaccardIndex(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                    'acc_oracle': Accuracy(num_classes=self.cfg.dataset_evaluation.num_classes, task="multiclass").to(self.device),
                }
    
    @torch.inference_mode()
    def get_prediction_single_model(self, model_idx, image):
        """Get softmax predictions from single model"""
        backbone = self.backbones[model_idx]
        classifier = self.classifiers[model_idx]
        
        lr_feats, _ = backbone(image)
        logits = classifier(lr_feats)
        if logits.shape[-2:] != [self.eval_img_size,self.eval_img_size]:
            logits = F.interpolate(logits, size=[self.eval_img_size,self.eval_img_size], mode="bilinear", align_corners=False)
        return logits
    
    def compute_majvote_ensemble(self, indices, all_preds):
        """Majority voting: most common prediction across selected models with random tie-breaking"""
        pred_stack = all_preds[list(indices)]  # (num_models_in_ensemble, B, H, W)

        # For ensembles of size 2 or when ties can occur, use random tie-breaking
        if len(indices) == 2:
            # When models disagree, randomly pick one
            # Generate random binary mask for tie-breaking
            agree_mask = pred_stack[0] == pred_stack[1]  # Where models agree
            random_choice = torch.randint(0, 2, pred_stack[0].shape, device=pred_stack.device)

            # Where they agree, use the agreed value; where they disagree, random choice
            maj_pred = torch.where(agree_mask, pred_stack[0],
                                   torch.where(random_choice == 0, pred_stack[0], pred_stack[1]))
        else:
            # For size > 2, use torch.mode (still has tie issues but less severe)
            maj_pred = torch.mode(pred_stack, dim=0)[0]  # (B, H, W)

        return maj_pred

    
    def compute_oracle_ensemble(self, indices, target, all_preds):
        """Oracle using pre-computed pred stack"""
        pred_stack = all_preds[list(indices)]  # (num_models_in_ensemble, B, H, W)
        correct_mask = (pred_stack == target.unsqueeze(0))
        oracle_pred = torch.where(
            correct_mask.any(dim=0),
            target,
            all_preds[indices[0]]  # Fallback to first model
        )
        return oracle_pred

    
    def reset_batch_predictions(self):
        """Clear stored predictions for next batch"""
        self.batch_predictions = {}
    
    @torch.inference_mode()
    def add_model_prediction(self, model_idx, batch, image_key="image"):
        """Process batch through single model and store its predictions"""
        batch = get_batch(batch, self.device)
        image = batch[image_key]
        
        logits = self.get_prediction_single_model(model_idx, image)
        self.batch_predictions[model_idx] = logits.argmax(dim=1)
        
        return batch
    
    def evaluate_ensemble_metrics(self, batch):
        """Given all model predictions in batch_predictions, evaluate ensembles"""
        # Extract target and valid mask (same for all models)
        target = batch["label"].to(self.device)
        valid_mask = target != 255
        
        # Verify all models have made predictions
        assert len(self.batch_predictions) == self.num_models, \
            f"Expected {self.num_models} predictions, got {len(self.batch_predictions)}"
        
        all_preds = torch.stack([self.batch_predictions[i] for i in range(self.num_models)], dim=0)
    
        # Process each ensemble configuration
        for ensemble_key in self.ensemble_metrics.keys():
            # Compute majvote ensemble
            majvote_pred = self.compute_majvote_ensemble(ensemble_key, all_preds)
            majvote_flat = rearrange(majvote_pred, "b h w -> (b h w)")
            
            # Compute oracle ensemble
            oracle_pred = self.compute_oracle_ensemble(ensemble_key, target, all_preds)
            oracle_flat = rearrange(oracle_pred, "b h w -> (b h w)")
            
            # Prepare target and mask
            target_flat = rearrange(target, "b h w -> (b h w)")
            valid_mask_flat = rearrange(valid_mask, "b h w -> (b h w)")
            
            # Filter by valid mask
            majvote_valid = majvote_flat[valid_mask_flat]
            oracle_valid = oracle_flat[valid_mask_flat]
            target_valid = target_flat[valid_mask_flat]
            
            # Update metrics
            self.ensemble_metrics[ensemble_key]['iou_majvote'].update(majvote_valid, target_valid)
            self.ensemble_metrics[ensemble_key]['acc_majvote'].update(majvote_valid, target_valid)
            self.ensemble_metrics[ensemble_key]['iou_oracle'].update(oracle_valid, target_valid)
            self.ensemble_metrics[ensemble_key]['acc_oracle'].update(oracle_valid, target_valid)
    
    def compute_results(self):
        """Compute final metrics for all ensembles"""
        results = {}
        for ensemble_key in sorted(self.ensemble_metrics.keys()):
            ensemble_name = f"ensemble_{'_'.join(map(str, ensemble_key))}"
            results[ensemble_name] = {
                'mIoU_majvote': self.ensemble_metrics[ensemble_key]['iou_majvote'].compute().item(),
                'accuracy_majvote': self.ensemble_metrics[ensemble_key]['acc_majvote'].compute().item(),
                'mIoU_oracle': self.ensemble_metrics[ensemble_key]['iou_oracle'].compute().item(),
                'accuracy_oracle': self.ensemble_metrics[ensemble_key]['acc_oracle'].compute().item(),
            }
        return results
    
    def save_results_csv(self, results, output_path):
        """Save results to CSV"""
        rows = []
        for ensemble_name, metrics in results.items():
            row = {'ensemble': ensemble_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return df


def main(cfg):
    print("\n CONFIGS \n",cfg, "\n CONFIGS \n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load backbone configs from ensemble config
    backbone_names = cfg.ensemble.backbones

    print(f"Loading {len(backbone_names)} backbones: {backbone_names}")
    backbones = []
    head_paths = []
    dataloaders = []
    for bb_name in backbone_names:
        backbone_cfg = OmegaConf.load(os.path.join(cfg.ensemble.backbone_dir,f"{bb_name}.yaml"))
        backbone = instantiate(backbone_cfg)
        backbones.append(backbone)
        backbone_id_name = backbone.get_identifiable_name()
        head_paths.append(os.path.join(cfg.ensemble.head_dir,f"{backbone_id_name}-{cfg.eval.upsampler}-seg-classifier.pth"))
        # Get dataloader with backbone-specific transforms
        _, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True)
        # print(val_loader)
        dataloaders.append(val_loader)
        print(f"  Loaded backbone: {bb_name}")
    print(f"Loaded {len(dataloaders)} validation dataloaders with backbone-specific transforms\n")
    
    # Initialize ensemble evaluator
    evaluator = EnsembleEvaluator(backbones, head_paths, device, cfg)
    
    # Evaluate on validation set
    # Zip all dataloaders and iterate through synchronized batches
    print(f"Evaluating {evaluator.num_models} models and {len(evaluator.ensemble_metrics)} ensemble combinations...")
    
    dataloader_iters = [iter(dl) for dl in dataloaders]
    num_batches = len(dataloaders[0])
    for batches_list in tqdm(zip_longest(*dataloader_iters), total=num_batches, desc="Evaluating"):
        if any(b is None for b in batches_list):
            break
        evaluator.reset_batch_predictions()
        # Get predictions from each model on its backbone-specific batch
        batch_with_target = None
        for model_idx, batch in enumerate(batches_list):
            batch_with_target = evaluator.add_model_prediction(model_idx, batch)
        
        # Evaluate all ensemble combinations with accumulated predictions
        evaluator.evaluate_ensemble_metrics(batch_with_target)
    
    # Compute and save results
    results = evaluator.compute_results()
    
    # Print results
    print("\n" + "="*100)
    print("Ensemble Evaluation Results")
    print("="*100)
    for ensemble_name, metrics in results.items():
        print(f"\n{ensemble_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save to CSV
    output_dir = os.path.join(cfg.project_root, cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ensemble_evaluation.csv")
    df = evaluator.save_results_csv(results, output_path)
    
    print("\n" + "="*100)
    return df


if __name__ == "__main__":
    import hydra
    
    @hydra.main(config_path="../config", config_name="eval_ensemble", version_base=None)
    def run(cfg):
        main(cfg)
    
    run()