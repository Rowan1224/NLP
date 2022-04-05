#!/bin/bash

#SBATCH --job-name=albert-fine_tune_model
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=m.z.hossain@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module load cuDNN
module load Python/3.9.5-GCCcore-10.3.0
cd $HOME/NLP
python3 -m venv env
source ./env/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

for i in 4 8 16;
do
    for j in 1e-5 3e-5 5e-5;
    do 
        python3 fine_tune.py -m albert -t base -b $i -lr $j
    done
done


