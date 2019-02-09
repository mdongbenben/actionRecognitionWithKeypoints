#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=121000
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module add opencv/3.3.0-v2
#module add cuda/9.0
#module add cudnn/7-cuda-9.0


mkdir -p /ssd_scratch/cvit/aryaman.g 
mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets
cd /ssd_scratch/cvit/aryaman.g/action_recognition_datasets
if [ ! -e "ssd_scratch/cvit/aryaman.g/action_recognition_datasets/UCF101.rar" ]
then
    wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
    unrar e -r UCF101.rar
    mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/UCF-101-videos 
    mv /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/*.avi /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/UCF-101-videos/
    wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
    unzip UCF101TrainTestSplits-RecognitionTask.zip
fi
mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames
mkdir -p /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames

python /home/aryaman.g/projects/humanActionRecognition/withoutSelectNode/genFramesTrainUCF.py
python /home/aryaman.g/projects/humanActionRecognition/withoutSelectNode/genFramesTestUCF.py

cd /home/aryaman.g/projects/openpose
module unload openblas/0.2.20
module load opencv/3.3.0
#module load cuda/8.0
export LD_LIBRARY_PATH=/usr/local/apps/cuDNN/5.1/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/apps/cuDNN/5.1/include:$CPATH
echo $LD_LIBRARY_PATH
echo $PATH
#module load cudnn/5.1-cuda-8.0

if [ ! -d "/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames_keypoint/" ] 
then
    echo "openpose not ran yet on train video frames. running now:" 
    ./build/examples/openpose/openpose.bin --display 0 --net_resolution "320x240" --image_dir /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames/ --write_keypoint /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames_keypoint/
fi

if [ ! -d "/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames_keypoint/" ] 
then
    echo "openpose not ran yet on test video frames. running now:" 
    ./build/examples/openpose/openpose.bin --display 0 --net_resolution "320x240" --image_dir /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames/ --write_keypoint /ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames_keypoint/
fi
cd /home/aryaman.g/projects/humanActionRecognition
