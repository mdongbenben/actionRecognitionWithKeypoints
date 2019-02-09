from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from ucfDataLoader import ucfDataLoad
from simpleNet import simpNet
from largeNet import largeNet
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
NB_EPOCHS = 256*2
fer = np.zeros((5,3))
trainFile = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/trainlist01.txt'
testFile = '/ssd_scratch/cvit/aryaman.g/action_recognition_datasets/ucfTrainTestlist/testlist01.txt'
criterion = nn.CrossEntropyLoss()
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
            #print(data.shape)
            output = model(data)
            #print(output)
            #loss = loss_func(output, target)  
            loss = criterion(output, target)
            loss.backward() 
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        elapsed = time.time() - t
        print (' -%d-  Epoch [%d/%d]'%(elapsed, epoch+1, NB_EPOCHS))
        test()

def test(prnt=False, it=-1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            #loss = loss_func(output, target)     # must be (1. nn output, 2. target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(correct, len(test_loader.dataset))
        print('Test Accuracy:', correct/len(test_loader.dataset)*100)

for itr in range(0,1):
    print("kth fold iter:", itr)
    train_dataset = ucfDataLoad(trainFile,1)
    test_dataset = ucfDataLoad(testFile,0)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = largeNet().to(device)
    #optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr = 0.00005)
    loss_func = torch.nn.MSELoss()
    train(NB_EPOCHS)
    test(prnt=True, it=itr)

