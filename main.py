# -*- coding:utf-8 -*-
import cv2
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import warnings
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

warnings.filterwarnings("ignore")

# gpu_id = 1
# torch.cuda.set_device(gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset = '/home/jinHM/liziyi/Protein/dataset/splited/'
BATCHSIZE = 112
IMAGE_DIMS = (224, 224, 3)
LEARNING_RATE = 0.001  # 学习率的设置
MODEL_NAME = 'resnet50_0624_2'


# 定义读取文件的格式
def default_loader(imagepath):
    # return Image.open(path).convert('RGB')
    image = cv2.imread(imagepath)
    image = cv2.resize(image, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
    image = image.astype("float")
    image = img_to_array(image)
    return image


class MyDataset(Dataset):
    def __init__(self, root, csv, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        df = pd.read_csv(csv)
        labels = []
        files = []
        for row in df.iterrows():
            filename = row[1]['filename']
            files.append(os.path.join(root, filename))
            labels.append(row[1]['label'].split(';'))
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        # loop over each of the possible class labels and show them
        print('[INFO]: {} classes found'.format(len(mlb.classes_)))
        # for (i, label) in enumerate():
        #     print("{}. {}".format(i + 1, label))

        self.imgs = list(zip(files, labels))
        print('[INFO]: {} images found'.format(len(self.imgs)))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        img = self.loader(fn)
        # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)
            # 数据标签转换为Tensor
        return img, label
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def train_and_valid(model, loss_function, optimizer, epochs, datasize):
    train_data_size, valid_data_size = datasize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = []
    best_auc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_auc = 0.0
        train_macro_f1 = 0.0
        train_micro_f1 = 0.0

        valid_loss = 0.0
        valid_auc = 0.0
        valid_macro_f1 = 0.0
        valid_micro_f1 = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            # print('labels:', labels)

            outputs = model(inputs)
            # print("outputs:", outputs)

            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            pred = torch.round(outputs.data).cpu().numpy()
            ground_truth = labels.cpu().numpy()
            proba = outputs.data.cpu().numpy()

            try:
                auc = roc_auc_score(ground_truth, proba, average='macro')
                train_auc += auc.item() * inputs.size(0)
            except ValueError:
                pass
            macro_f1 = f1_score(ground_truth, pred, average='macro')
            micro_f1 = f1_score(ground_truth, pred, average='micro')
            # print(macro_f1, micro_f1)

            train_macro_f1 += macro_f1.item() * inputs.size(0)
            train_micro_f1 += micro_f1.item() * inputs.size(0)

        # print(train_loss, train_auc, train_macro_f1, train_micro_f1)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                outputs = model(inputs)
                # print("outputs:", outputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                pred = torch.round(outputs.data).cpu().numpy()
                ground_truth = labels.cpu().numpy()
                proba = outputs.data.cpu().numpy()

                try:
                    auc = roc_auc_score(ground_truth, proba, average='macro')
                    valid_auc += auc.item() * inputs.size(0)
                except ValueError:
                    pass
                macro_f1 = f1_score(ground_truth, pred, average='macro')
                micro_f1 = f1_score(ground_truth, pred, average='micro')

                valid_macro_f1 += macro_f1.item() * inputs.size(0)
                valid_micro_f1 += micro_f1.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_auc = train_auc / train_data_size
        avg_train_macro_f1 = train_macro_f1 / train_data_size
        avg_train_micro_f1 = train_micro_f1 / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_auc = valid_auc / valid_data_size
        avg_valid_macro_f1 = valid_macro_f1 / valid_data_size
        avg_valid_micro_f1 = valid_micro_f1 / valid_data_size

        history.append([avg_train_loss, avg_valid_loss,
                        avg_train_auc, avg_valid_auc,
                        avg_train_macro_f1, avg_valid_macro_f1,
                        avg_train_micro_f1, avg_valid_micro_f1])

        if best_auc < avg_valid_auc:
            best_auc = avg_valid_auc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:02d}, Training: loss: {:.2f}, auc: {:.2f}%, macro_f1: {:.2f}%, micro_f1: {:.2f}%".format(
            epoch + 1,
            avg_train_loss,
            avg_train_auc * 100,
            avg_train_macro_f1 * 100,
            avg_train_micro_f1 * 100))

        print("\t\tValidation: loss: {:.2f}, auc: {:.2f}%, macro_f1: {:.2f}%, micro_f1: {:.2f}%, Time: {:.2f}s".format(
            avg_valid_loss,
            avg_valid_auc * 100,
            avg_valid_macro_f1 * 100,
            avg_valid_micro_f1 * 100,
            epoch_end - epoch_start))

        print("Best auc for validation : {:.2f}% at epoch {:02d}".format(best_auc * 100, best_epoch))

        print('[INFO]: serializing model...')
        torch.save(model,
                   '/home/jinHM/liziyi/Protein/Torch_Train/models/' + MODEL_NAME + '_model_' + str(epoch + 1) + '.pt')
        print('-' * 70)

    print('[INFO]: saving history...')
    torch.save(history, '/home/jinHM/liziyi/Protein/Torch_Train/models/' + MODEL_NAME + '_history.pt')

    # return model


if __name__ == '__main__':
    train_data = MyDataset(root='/home/jinHM/liziyi/Protein/dataset/splited/train', csv=dataset + 'train.csv',
                           transform=transforms.ToTensor())
    # test_data = MyDataset(root='/home/jinHM/liziyi/Protein/dataset/splited/test', csv=dataset + 'test.csv',
    #                       transform=transforms.ToTensor())
    valid_data = MyDataset(root='/home/jinHM/liziyi/Protein/dataset/splited/valid', csv=dataset + 'valid.csv',
                           transform=transforms.ToTensor())

    train_data_size = len(train_data)
    # test_data_size = len(test_data)
    valid_data_size = len(valid_data)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=6)
    # test_loader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False, num_workers=6)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCHSIZE, shuffle=False, num_workers=6)

    resnet50 = models.resnet50(pretrained=False)

    # for param in resnet50.parameters():
    #     param.requires_grad = False

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        # nn.LogSoftmax(dim=1)
        nn.Sigmoid()
    )
    resnet50 = resnet50.to('cuda:0')

    loss_func = nn.BCELoss()
    optimizer = optim.Adam(resnet50.parameters())

    train_and_valid(resnet50, loss_func, optimizer, epochs=30, datasize=(train_data_size, valid_data_size))
