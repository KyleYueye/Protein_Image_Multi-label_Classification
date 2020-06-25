import numpy as np
import os
import shutil
from tqdm import tqdm

test_path = '/home/jinHM/liziyi/Protein/dataset/splited/test'
valid_path = '/home/jinHM/liziyi/Protein/dataset/splited/valid'
train_path = '/home/jinHM/liziyi/Protein/dataset/splited/train'


def transfer():
    test_list = os.listdir(test_path)
    size = len(test_list)

    np.random.shuffle(test_list)

    new_test_list = test_list[:size // 2]
    valid_list = test_list[size // 2:]

    for img in tqdm(valid_list):
        oldpos = os.path.join(test_path, img)
        newpos = os.path.join(valid_path, img)
        shutil.move(oldpos, newpos)


def build_csv():
    # test_list = os.listdir(test_path)
    # valid_list = os.listdir(valid_path)
    train_list = os.listdir(train_path)
    f = open('/home/jinHM/liziyi/Protein/dataset/splited/train.csv', 'w')
    f.write('filename,label\n')
    for img in tqdm(train_list):
        with open('/home/jinHM/liziyi/Protein/dataset/image_label_dir/{}.txt'.format(img), 'r') as n:
            label = [x.strip() for x in n.readlines()]
            label.sort()
        f.write('{},{}\n'.format(img, ';'.join(label)))
    f.close()


if __name__ == '__main__':
    # transfer()
    build_csv()
