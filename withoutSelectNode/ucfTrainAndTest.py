from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ucfDataLoader import ucfDataLoad
from simpleNet import simpNet
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random      
import time
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
BATCH_SIZE = 32
NB_EPOCHS = 100
fer = np.zeros((5,3))
trainFile = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/trainlist01.txt'
testFile = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/testlist01.txt'
"""
with open(origFile,'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open(shuffleFile,'w') as target:
    for _, line in data:
        target.write( line )
"""
def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def train(epochs):
    model.train()
    for epoch in range(0,epochs):
        train_loss = 0
        t = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   # clear gradients for next train
            output = model(data)
            loss = loss_func(output, target)  
            train_loss += loss*output.shape[0]*3   # must be (1. nn output, 2. target)
            loss.backward() 
            optimizer.step()
        train_loss = train_loss/(len(train_loader.dataset)*3) 
        elapsed = time.time() - t
        print (' -%d-  Epoch [%d/%d]'%(elapsed, epoch+1, NB_EPOCHS))
        print ('Training samples: %d Train Loss: %.5f'%(len(train_loader.dataset), train_loss.item()))
        test()
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def test(prnt=False, it=-1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        yer = 0
        per = 0
        rer = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)     # must be (1. nn output, 2. target)
            test_loss += loss*3*output.shape[0]
            if prnt:
                for i in range(output.shape[0]):
                    yer += abs(output[i][0]-target[i][0])
                    per += abs(output[i][1]-target[i][1])
                    rer += abs(output[i][2]-target[i][2])
    test_loss /= (len(test_loader.dataset)*3)
    print ('Test samples: %d Test Loss: %.5f'%(len(test_loader.dataset), test_loss.item()))
    if prnt:
        er1 = ((yer.item())/len(test_loader.dataset))*180
        er2 = ((per.item())/len(test_loader.dataset))*180
        er3 = ((rer.item())/len(test_loader.dataset))*180
        fer[it-1][0] = er1
        fer[it-1][1] = er2
        fer[it-1][2] = er3
        print ('Mean Absolute Error: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(er1, er2, er3, (er1+er2+er3)/3))

for itr in range(0,1):
    print("kth fold iter:", itr)
    train_dataset = retDataset2(trainFile,1,itr)
    test_dataset = retDataset2(testFile,0,itr)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = simpNet().to(device)
    #optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    loss_func = torch.nn.MSELoss()
    train(NB_EPOCHS)
    test(prnt=True, it=itr)

er1 = 0
er2 = 0
er3 = 0
for i in range(0,5):
    er1 += fer[i][0]
    er2 += fer[i][1]
    er3 += fer[i][2]
er1 = er1/5
er2 = er2/5
er3 = er3/5
print('5 fold MAE: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(er1, er2, er3, (er1+er2+er3)/3))
