import numpy as np
import cv2
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import acnet

#use gpu whenever available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_ckp(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def trans(image, m1, m2, m3, m4, m5):
    #resize the test images
    resize = transforms.Resize(size=(256, 256))
    image = resize(image)
    resize1 = transforms.Resize(size=(128, 128))
    m1 = resize1(m1)
    m2 = resize1(m2)
    m3 = resize1(m3)
    m4 = resize1(m4)
    m5 = resize1(m5)
    #convert to tensor
    to_T = transforms.ToTensor()
    image = to_T(image)
    m1 = to_T(m1)
    m2 = to_T(m2)
    m3 = to_T(m3)
    m4 = to_T(m4)
    m5 = to_T(m5)
    masks = torch.cat((m1, m2, m3, m4, m5), dim=0)
    #normalize the input image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    return image, masks

def load_data(ID):
    image = Image.open(ID)
    m_path = "/home/indranil/action_recognition/test_masks/"
    m1 = None
    m2 = None
    m3 = None
    m4 = None
    m5 = None
    for i in range(5):
        m_index = int((ID.split("/")[-1]).split(".")[0])
        m_index -= i
        if m_index < 0:
            m_index = 0
        m_ID = m_path + str(m_index) + ".png"
        m_img = Image.open(m_ID).convert('L')
        if i==0:
            m1 = m_img
        if i==1:
            m2 = m_img
        if i==2:
            m3 = m_img
        if i==3:
            m4 = m_img
        if i==4:
            m5 = m_img
    image, masks = trans(image, m1, m2, m3, m4, m5)
    return image, masks

def main():
    #prepare images and labels list
    images_path = "/home/indranil/action_recognition/test_img/"
    mask_path = "/home/indranil/action_recognition/test_masks"
    output_path = "/home/indranil/action_recognition/out_frames/"
    
    #create and load the pretrained refinenet model
    net = acnet.acnet().to(device)
    last_model_path = "/home/indranil/action_recognition/models/epoch_79.pt"
    net = load_ckp(last_model_path, net)
    net.eval()

    dirs = os.listdir(images_path)
    for item in dirs:
        image_item = images_path + item
        if os.path.isfile(image_item):
            with torch.no_grad():
                out_item = output_path + item.split(".")[0] + ".png"
                X1, X2 = load_data(image_item)
                X1 = torch.unsqueeze(X1,0)
                X2 = torch.unsqueeze(X2,0)
                y = net(X1.to(device), X2.to(device))
                y = torch.squeeze(y)
                _, max_index = torch.topk(y, 3, dim=0)
                max_index = np.asarray(max_index.detach().to('cpu'))
                # print(max_index[0])
                ori_image = cv2.imread(image_item)
                if max_index[0] == 2:
                    cv2.putText(ori_image,'Stirring',(400,200), cv2.FONT_HERSHEY_SIMPLEX, 3,(8,191,145),3)
                if max_index[0] == 1:
                    cv2.putText(ori_image,'Adding ingredients',(120,200), cv2.FONT_HERSHEY_SIMPLEX, 3,(8,191,145),3)
                cv2.imwrite(out_item, ori_image)

    return


if __name__ == "__main__":
    main()