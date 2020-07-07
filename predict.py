import os
import random
from time import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from imutils import paths
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def image_loader(img, size):
    image = cv2.imread(img)
    tran = transforms.ToTensor()
    image = cv2.resize(image, size)
    image = image.astype("float")
    image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    image = tran(image)
    image = image.unsqueeze(0)
    return image


class Predict:
    def __init__(self, modelPath, size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = modelPath
        self.dataset = '/home/jinHM/liziyi/Protein/dataset/splited/test/'
        self.label_path = '/home/jinHM/liziyi/Protein/dataset/image_label_dir/'
        tik = time()
        print("[INFO] loading network...")
        model = torch.load(self.model_path)
        self.model = model.to(self.device)
        self.model.eval()
        tok = time()
        dur = (tok - tik)
        print("{:.1f}s in loading network".format(dur))

        self.imagePaths = list(paths.list_images(self.dataset))
        self.length = len(self.imagePaths)
        self.size = size

    def random_predict(self):
        imagepath = self.imagePaths[random.randint(0, self.length)]
        img = imagepath.split(os.sep)[-1]
        print('-' * 35)
        print(img)
        with open(os.path.join(self.label_path, img) + '.txt') as f:
            labels = [int(x.strip()) for x in f.readlines()]

        image = image_loader(imagepath, self.size)
        image = image.to(self.device)
        with torch.no_grad():
            tensor = self.model(image)

        proba = list(tensor.cpu().numpy()[0])

        idxs = np.argsort(proba)[::-1]
        for i in idxs:
            if i in labels:
                print('* Class {}: {:.2f}%'.format(i, proba[i] * 100))
            else:
                print('  Class {}: {:.2f}%'.format(i, proba[i] * 100))

    def specific_predict(self, img):
        print(img)
        with open(os.path.join(self.label_path, img) + '.txt') as f:
            labels = [int(x.strip()) for x in f.readlines()]

        image = image_loader(imagepath, self.size)
        image = image.to(self.device)

        with torch.no_grad():
            tensor = self.model(image)

        proba = list(tensor.cpu().numpy()[0])

        idxs = np.argsort(proba)[::-1]
        for i in idxs:
            if i in labels:
                print('* Class {}: {:.2f}%'.format(i, proba[i] * 100))
            else:
                print('  Class {}: {:.2f}%'.format(i, proba[i] * 100))

    def overall_eval(self):
        # w = open('/home/jinHM/liziyi/Protein/dataset/splited/test_result.csv', 'w')
        y_true = []
        y_pred = []
        print("[INFO] reading and predicting images...")
        tik = time()
        for imagepath in tqdm(self.imagePaths):
            img = imagepath.split(os.sep)[-1]
            with open(os.path.join(self.label_path, img) + '.txt') as f:
                labels = [int(x.strip()) for x in f.readlines()]

            y_true.append(labels)
            image = image_loader(imagepath, self.size)
            image = image.to(self.device)

            with torch.no_grad():
                tensor = self.model(image)
            proba = list(tensor.cpu().numpy()[0])
            y_pred.append(proba)
        tok = time()
        dur = (tok - tik)
        print("{:.1f}s in loading images".format(dur))
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)

        print("[INFO] computing metrics...")
        gt, prob = np.array(y_true), np.array(y_pred)
        pred = np.round(y_pred)
        macro_f1 = f1_score(gt, pred, average='macro')
        micro_f1 = f1_score(gt, pred, average='micro')
        roc_auc = roc_auc_score(gt, prob, average="macro")
        print('macro_f1 = {:.2f}%'.format(macro_f1 * 100))
        print('micro_f1 = {:.2f}%'.format(micro_f1 * 100))
        print('roc_auc = {:.2f}%'.format(roc_auc * 100))

    def predict_on_test(self, root_path):
        print("[INFO] reading and predicting images...")
        tik = time()
        output = []
        dir_list = os.listdir(root_path)
        for dir in tqdm(dir_list):
            store = []
            img_list = os.listdir(os.path.join(root_path, dir))
            for img in img_list:
                image = image_loader(os.path.join(root_path, dir, img), self.size)
                image = image.to(self.device)

                with torch.no_grad():
                    tensor = self.model(image)
                proba = list(tensor.cpu().numpy()[0])
                store.append([img, proba, np.round(proba)])
            output.append([dir, store])
        tok = time()
        dur = (tok - tik)
        print("{:.1f}s in loading images".format(dur))
        np.save('/home/jinHM/liziyi/Protein/dataset/final_test.npy', output)
        print("[INFO]: Done")


if __name__ == '__main__':
    resnet_50_path = '/home/jinHM/liziyi/Protein/Torch_Train/models/0624_1/resnet50_0624_1_model_15.pt'
    resnet_50_path_30epochs = '/home/jinHM/liziyi/Protein/Torch_Train/models/0624_2/resnet50_0624_2_model_24.pt'
    pred = Predict(resnet_50_path_30epochs, (224, 224))

    pred.predict_on_test('/home/jinHM/liziyi/Protein/dataset/final_test')

    # pred.overall_eval()

    # for i in range(30):
    #     pred.random_predict()

    # pred.specific_predict('964_E9_1_blue_red_green.jpg_19.jpg')
