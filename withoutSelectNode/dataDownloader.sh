#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=11000
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
#module add opencv/3.3.0
#module add cuda/9.0
#module add cudnn/7-cuda-9.0


mkdir -p /ssd_scratch/cvit/aryaman.g 
mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets
cd /ssd_scratch/cvit/aryaman.g/action_recognition_datasets
wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e -r UCF101.rar
wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/

