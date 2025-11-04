#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/tz_train_probes%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

UPSAMPLER=none # bilinear, anyup, none
BACKBONE=radio # base (vit_small), radio, dinov2
ADAPTOR=backbone # backbone, dino_v2, sam, siglip, clip
module load python/3.10
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip -q
pip install -r requirements.txt -q

# For RADIO
python -u evaluation/tz_train_probes.py backbone=$BACKBONE backbone.name=radio_v2.5-b  eval.upsampler=$UPSAMPLER backbone.adaptor_names=$ADAPTOR

# For DINO
# python -u evaluation/tz_train_probes.py backbone=$BACKBONE backbone.name=dinov2_vitb14_reg eval.upsampler=$UPSAMPLER

# sbatch evaluation/tz_train_probes.sh