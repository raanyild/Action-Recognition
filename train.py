import numpy as np
import cv2
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataloader
import acnet


#use gpu whenever available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#saving and loading model checkpoints
def save_ckp(state, path):
    torch.save(state, path)
    return 

def load_ckp(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def train(net, train_loader, criterion, optimizer, start_epoch=0, end_epoch=0, model_path=None, log_path=None):
    experiment_name = 'acnet'
    writer = SummaryWriter(log_path + experiment_name)
    iteration = 0
    for epoch in range(start_epoch, end_epoch):
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            input1, input2, label = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(input1, input2)
            loss = criterion(outputs, (label.long()).squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("Epoch {0} & Iteration {1} : current loss {2}".format(epoch, iteration, loss.item()))
            iteration += 1

        total_loss /= len(train_loader)
        writer.add_scalar('training_loss', total_loss, epoch)

        if(epoch%5 == 4):
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            ckp_path = model_path + "epoch_" + str(epoch) + ".pt"
            save_ckp(checkpoint, ckp_path)
            
    return
       

def main():
    #prepare images and labels list
    images_path = "/home/indranil/action_recognition/test_img/"
    label_path = "/home/indranil/action_recognition/labels/"
    images_id = []
    labels_map = {}
    dirs = os.listdir(images_path)
    for item in dirs:
        image_item = images_path + item
        if os.path.isfile(image_item):
            label_item = label_path + item.split(".")[0] + ".txt"
            if os.path.isfile(label_item):
                images_id.append(image_item)
                labels_map[image_item] = label_item

    #train data loader
    train_loader = dataloader.dataLoader(images_id, labels_map, batch_size = 16)

    #create the refinenet model
    net = acnet.acnet().to(device)

    # train the network epoch 1-100
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.6, 0.999), eps=1e-08)
    criterion = nn.CrossEntropyLoss().to(device)
    train(net=net, train_loader=train_loader, criterion=criterion, optimizer=optimizer, start_epoch=0, end_epoch=100,
        model_path="/home/indranil/action_recognition/models/", log_path="/home/indranil/action_recognition/log_dir/")
    return

if __name__ == "__main__":
    main()