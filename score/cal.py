import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

test_data = []
with open('test.csv', 'r') as test:
    for line in test.readlines():
        line = line.strip()
        folder, label = line.split(',')
        label_list = label.split(';')
        test_data.append(label_list)

mlb = MultiLabelBinarizer()
test_data = mlb.fit_transform(test_data)

test_pred_data = []
with open('test-pred.csv', 'r') as test_pred:
    for line in test_pred.readlines():
        line = line.strip()
        folder, label_pred = line.split(',')
        pred_list = label_pred.split(';')
        for i in range(len(pred_list)):
            pred_list[i] = eval(pred_list[i])
        test_pred_data.append(pred_list)

true_list = []
with open('true_test.csv', 'r') as true:
    for line in true.readlines():
        line = line.strip()
        folder, label = line.split(':')
        label_list = label.split(';')
        true_list.append(label_list)

true_list = mlb.fit_transform(true_list)

macro_f1 = f1_score(true_list, test_data, average='macro')
micro_f1 = f1_score(true_list, test_data, average='micro')
roc_auc = roc_auc_score(true_list, np.array(test_pred_data), average='macro')

print('macro_f1 = {:.2f}%'.format(macro_f1 * 100))
print('micro_f1 = {:.2f}%'.format(micro_f1 * 100))
print('roc_auc = {:.2f}%'.format(roc_auc * 100))
