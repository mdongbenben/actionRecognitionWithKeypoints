from __future__ import print_function, division
import os,io
import torch
import numpy as np
import PIL
from PIL import Image
import cv2
import yaml
import math
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from skimage.transform import resize

class ucfDataLoad(Dataset):

    def __init__(self, path_file, trnFlg=0, transform=None):
        f = open(path_file,"r")
        self.alines = f.readlines()
        self.transform = transform
        self.trnFlg = trnFlg
        self.allVideos = []
        self.allgt = []
        self.allsn = []
        min_frames = 1000000
        wrong_samples = []
        diff_samples = []
        h = open('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/classInd.txt' ,"r")
        action_class_mapping = {}
        aclines = h.readlines()
        for ln in aclines:
            action_class_mapping[ln.split()[1]] = int(ln.split()[0]) 
        if self.trnFlg == 1:
            g = open('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/train_frame_count.txt',"w")
            frames_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames/'
            keypoints_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/all_ucf_train_video_frames_keypoint/'
        else:
            g = open('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/test_frame_count.txt',"w")
            frames_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames/'
            keypoints_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/all_ucf_test_video_frames_keypoint/'
        for idx in range(0,len(self.alines)):
                line = self.alines[idx]
                line = line.split()
                sample_name = line[0].split("/")[-1]
                if self.trnFlg == 0:
                    action_name = line[0].split("/")[0]
                    label = action_class_mapping[action_name]
                else:
                    label = int(line[1])
                print(sample_name)
                sample_name_without_extension = sample_name.split(".")[0]
                cap = cv2.VideoCapture('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/UCF-101-videos/'+sample_name)
                video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
                video_frame_count = 0 
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        sample_name_without_extension = sample_name.split(".")[0]
                        video_frame_name = sample_name_without_extension + "_{0:0=5d}".format(video_frame_count) +".jpg"
                        video_frame_path = frames_path 
                        #print(video_frame_name)
                        cv2.imwrite(os.path.join(video_frame_path, video_frame_name), frame)  # save frame as JPEG file
                        video_frame_count += 1
                        #break
                    else:
                        break
                g.write(sample_name+" "+ str(video_frame_count) + "\n")
                #print("min frames uptil now: ",  min_frames)
                #if video_frame_count != video_length:
                #    wrong_samples.append(sample_name)
                #    diff_samples.append(video_frame_count - video_length)
                #print("frames in this video: ",  video_frame_count == video_length)
        #for i in range(0, len(wrong_samples)):
        #    print(wrong_samples[i], diff_samples[i])
        f.close()
        g.close()
        h.close()
    def __getitem__(self, idx):
        hm = self.allVideos[idx]
        lbl = self.allgt[idx]
        video = torch.from_numpy(hm)
        func_choice = random.choice([True, False])
        if func_choice and self.trnFlg==1:
            pass 
        lbl =  torch.from_numpy(lbl)
        if self.transform:
            video = self.transform(video)
        return video, lbl
	
    def __len__(self):
        #return 10
        if self.trnFlg==1:
            return len(self.allVideos)
            #return 2512 
        return len(self.allVideos)

    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]
train_dataset = ucfDataLoad('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/testlist01.txt',0)
#train_dataset = retDataset('/home/aryaman.g/pyTorchLearn/biwiTrain.txt',1)
#train_dataset = biwiDataset('/ssd_scratch/cvit/aryaman.g/biwiHighResHM/allHM',1)
#print(train_dataset.__getitem__(2))
