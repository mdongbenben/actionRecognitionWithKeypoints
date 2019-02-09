#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gnode24
#SBATCH --mem=121000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module load cuda/9.0
module load cudnn/7-cuda-9.0

#python3 ucfDataLoader.py
python3 ucfTrainAndTest.py
