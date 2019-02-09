from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from aflwDataLoader import retDataset2
from dataLoaderPyTorch import retDataset
from simpleNet import simpNet
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
BATCH_SIZE = 32
NB_EPOCHS = 200

train_dataset = retDataset2('/home/aryaman.g/pyTorchLearn/aflwTrain.txt')
test_dataset = retDataset2('/home/aryaman.g/pyTorchLearn/aflwTest.txt')
crossVal_dataset = retDataset('/home/aryaman.g/projects/cscFcPs/trainFCOR.txt')
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
crossVal_loader = DataLoader(dataset=crossVal_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


model = simpNet().to(device)
print(model)
#optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=1e-6)
#optimizer = optim.Adadelta(model.parameters(), rhs=0.95, eps=0)
#optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adam(model.parameters(), lr = 0.000007)
loss_func = torch.nn.MSELoss()

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
        print (' Train Loss: %.5f'%(train_loss.item()))
        test()
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def test(prnt=True):
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
    print('Test_loss: %.5f'%(test_loss.item()))
    if prnt:
        er1 = ((yer.item())/len(test_loader.dataset))*180
        er2 = ((per.item())/len(test_loader.dataset))*180
        er3 = ((rer.item())/len(test_loader.dataset))*180
        print ('Mean Absolute Error: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(er1, er2, er3, (er1+er2+er3)/3))
train(NB_EPOCHS)
test(prnt=True)
"""
def train(epochs):
    model.train()
    for epoch in range(0,epochs):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   # clear gradients for next train
            output = model(data)
            loss = loss_func(output, target)  
            train_loss += loss*output.shape[0]*3   # must be (1. nn output, 2. target)
            loss.backward() 
            optimizer.step()
        train_loss = train_loss/(len(train_loader.dataset)*3) 
        print("train_loss ",epoch,"/",NB_EPOCHS, train_loss.data)
        test()
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def test(prnt=False):
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
    print('test_loss ',test_loss.data)
    if prnt:
        print((yer/len(test_loader.dataset))*360, (per/len(test_loader.dataset))*360, (rer/len(test_loader.dataset))*360)

train(NB_EPOCHS)
test(prnt=True)
"""
lmdl = torch.load('aflw.pt')
torch.save(model,'aflw.pt')
lmdl.eval()
test_loss = 0
correct = 0
prnt=True
with torch.no_grad():
        yer = 0
        per = 0
        rer = 0
        for data, target in crossVal_loader:
            data, target = data.to(device), target.to(device)
            output = lmdl(data)
            loss = loss_func(output, target)     # must be (1. nn output, 2. target)
            test_loss += loss*3*output.shape[0]
            if prnt:
                for i in range(output.shape[0]):
                    for j in range(3):
                        output[i][j] = (output[i][j]-0.5)*2+0.5
                    yer += abs(output[i][0]-target[i][0])
                    per += abs(output[i][1]-target[i][1])
                    rer += abs(output[i][2]-target[i][2])
        if prnt:
            print((yer/len(crossVal_loader.dataset))*180, (per/len(crossVal_loader.dataset))*180, (rer/len(crossVal_loader.dataset))*180)
