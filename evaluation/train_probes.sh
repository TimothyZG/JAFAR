#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_probes%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module load python/3.10
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip -q
pip install -r requirements.txt -q

python -u evaluation/train_probes.py

# sbatch evaluation/train_probes.sh