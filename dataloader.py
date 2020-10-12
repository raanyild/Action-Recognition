import torch
import torchvision.transforms as transforms
from PIL import Image
import random

def transform(image, m1, m2, m3, m4, m5, is_train):
    if is_train == True:
        if random.random() > 0.6:    
            #colourJitter
            jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, hue=0.1, saturation=0.05)
            image = jitter(image)

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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, is_train = False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
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
        f = open(self.labels[ID], "r")
        label = int(f.read())
        image, masks = transform(image, m1, m2, m3, m4, m5, self.is_train)
        return image, masks, label


def dataLoader(training_images, labels, batch_size = 4):
    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8}
    train = Dataset(training_images, labels, True)
    train_loader = torch.utils.data.DataLoader(train, **params)
    return train_loader