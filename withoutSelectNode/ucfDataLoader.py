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
        if self.trnFlg == 1:
            g = open('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/train_frame_count.txt',"r")
            frames_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames/'
            keypoints_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/all_ucf_train_video_frames_keypoint/'
        else:
            g = open('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/train_frame_count.txt',"r")
            frames_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_test_video_frames/'
            keypoints_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/all_ucf_test_video_frames_keypoint/'
        self.fclines = g.readlines()        
        for idx in range(0,len(self.alines)):
                line = self.alines[idx]
                line = line.split()
                sample_name = line[0].split("/")[-1]
                label = int(line[1])
                print(sample_name)
                fcline = self.fclines[idx].split()
                video_frame_count = int(fcline[1])
                sample_name_without_extension = sample_name.split(".")[0]
                hm = np.zeros((27,54))
                #select 27 frames distributed equally over the video
                for s in range(0,27):
                    sel_frame_count = int(math.floor(video_frame_count/27))*s
                    keypoint_file = keypoints_path + sample_name_without_extension + "_{0:0=5d}".format(sel_frame_count) +"_pose.yml"
                    keypoint_file = keypoints_path + "v_ApplyEyeMakeup_g08_c01_00000_pose.yml"
                    
                    skip_lines = 3                    
                    with open(keypoint_file) as infile:
                        for i in range(skip_lines):
                            tmp = infile.readline()
                        data = yaml.load(infile)
                    rel_tol=1e-09
                    bodyPose = data['data']
                    if len(bodyPose)>0:
                        # if any person detected, select the largest person
                        spi = 0
                        mxa = 0
                        
                        def prob_retrieve(bodyPose, person_index, keypoint_index):
                            #select keypoint for person        
                            x = bodyPose[person_index*54+keypoint_index*3]
                            y = bodyPose[person_index*54+keypoint_index*3+1]
                            p = bodyPose[person_index*54+keypoint_index*3+2]
                            return x, y, p
                        # iterate over persons 
                        num_persons = int(len(bodyPose)/54)
                        for person_index in range(0, num_persons):
                            minx = 1000
                            miny = 1000
                            maxx = 0
                            maxy = 0

                            for keypoint_index in range(0,18):
                                [x, y, p] = prob_retrieve(bodyPose, person_index, keypoint_index):
                                if p!=0:
                                    minx = min(minx,x)
                                    miny = min(miny,y)
                                    maxx = max(maxx,x)
                                    maxy = max(maxy,y)
                            if (maxx-minx)*(maxy-miny)>mxa:
                                mxa = (maxx-minx)*(maxy-miny)
                                spi = person_index
                        
                        for keypoint_index in range(0,18):
                            [x, y, p] = prob_retrieve(bodyPose, spi, keypoint_index):
                            x = x/320
                            y = y/240
                            hm[s,keypoint_index*3] = x
                            hm[s,keypoint_index*3+1] = y
                            hm[s,keypoint_index*3+2] = p
                    break	
                break           
                self.allVideos.append(hm)
                self.allgt.append(label)
                #lbl = np.float32(np.asarray([-float(line[1])]))
                #cap = cv2.VideoCapture('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/UCF-101-videos/'+sample_name)
                #video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
                #video_frame_count = 0 
                #while(cap.isOpened()):
                #    ret, frame = cap.read()
                #    if ret == True:
                #        sample_name_without_extension = sample_name.split(".")[0]
                #        video_frame_name = sample_name_without_extension + "_{0:0=5d}".format(video_frame_count) +".jpg"
                #        video_frame_path = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucf_train_video_frames/' 
                #        #print(video_frame_name)
                #        cv2.imwrite(os.path.join(video_frame_path, video_frame_name), frame)  # save frame as JPEG file
                #        video_frame_count += 1
                #        #break
                #    else:
                #        break
                #min_frames = min(min_frames, video_frame_count)
                #f.write(sample_name+" "+ str(video_frame_count) + "\n")
                #print("min frames uptil now: ",  min_frames)
                #if video_frame_count != video_length:
                #    wrong_samples.append(sample_name)
                #    diff_samples.append(video_frame_count - video_length)
                #print("frames in this video: ",  video_frame_count == video_length)
                ##prts = nm.split("_")
                #namehm = prts[0] + '_' + prts[1] + '_rgb_c' + line[0].split("/")[-2] + '_heatmaps.png'
                #img = Image.open('/home/aryaman.g/projects/cscFcPs/allImgOut/'+namehm)
        #for i in range(0, len(wrong_samples)):
        #    print(wrong_samples[i], diff_samples[i])
        f.close()
        g.close()

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
    """
    def kfldShuff(self, ks=[]):
        self.Imgs = []
        self.gt = []
        self.kfold = ks
        if len(self.kfold)==0:
            self.Imgs = []
        self.gt = []
        self.kfold = ks
        if len(self.kfold)==0:
            for j in range(0,len(self.kfold)):
                if self.allsn[idx]==self.kfold[j]:
                    isPrs = True
            if self.trnFlg:
                isPrs = not isPrs
            if isPrs:
                self.Imgs.append(self.allImgs[idx])
                self.gt.append(self.allgt[idx]) 
    """   
train_dataset = ucfDataLoad('/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/trainlist01.txt',1,0)
#train_dataset = retDataset('/home/aryaman.g/pyTorchLearn/biwiTrain.txt',1)
#train_dataset = biwiDataset('/ssd_scratch/cvit/aryaman.g/biwiHighResHM/allHM',1)
#print(train_dataset.__getitem__(2))
