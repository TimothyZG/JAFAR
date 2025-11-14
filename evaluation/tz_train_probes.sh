#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/tz_train_probes%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export HYDRA_FULL_ERROR=1

UPSAMPLER=none # bilinear, anyup, none
module load python/3.10
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip -q
pip install -r requirements.txt -q

# For RADIO
ADAPTOR=clip # backbone, dino_v2, sam, siglip, clip
# python -u evaluation/tz_train_probes.py backbone=radio backbone.name=radio_v2.5-b  eval.upsampler=$UPSAMPLER backbone.adaptor_name=$ADAPTOR img_size=512

# For DINO v2
# python -u evaluation/tz_train_probes.py backbone=dinov2 backbone.name=dinov2-base eval.upsampler=$UPSAMPLER img_size=518

# For DINO v3
# python -u evaluation/tz_train_probes.py backbone=dinov3 backbone.name=dinov3-vitb16-pretrain-lvd1689m eval.upsampler=$UPSAMPLER img_size=512

# For SigLIP 
# python -u evaluation/tz_train_probes.py backbone=siglip backbone.name=siglip-base-patch16-512 eval.upsampler=$UPSAMPLER img_size=512

# For SigLIP v2
# python -u evaluation/tz_train_probes.py backbone=siglip backbone.name=siglip2-base-patch16-512 eval.upsampler=$UPSAMPLER img_size=512

# For Franca
# python -u evaluation/tz_train_probes.py backbone=franca backbone.name=franca_vitb14 backbone.weights=IN21K eval.upsampler=$UPSAMPLER img_size=518 # IN21K (b,l,g) or LAION (l,g)

# For DFN CLIP
# python -u evaluation/tz_train_probes.py backbone=dfnclip backbone.name=DFN2B-CLIP-ViT-B-16 eval.upsampler=$UPSAMPLER img_size=224

# For SAM
# python -u evaluation/tz_train_probes.py backbone=sam backbone.name=samvit_base_patch16.sa1b eval.upsampler=$UPSAMPLER img_size=512

# Evaluate Ensembles
python -u evaluation/tz_eval_ensembles.py

# sbatch evaluation/tz_train_probes.sh