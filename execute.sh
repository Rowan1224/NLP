#!/bin/bash

#SBATCH --job-name=nlp_fine_tune_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=d.a.tran@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB

module load cuDNN
module load Python/3.9.5-GCCcore-10.3.0
cd $HOME/NLP
python3 -m venv env
source ./env/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

python3 fine_tune_distil.py
