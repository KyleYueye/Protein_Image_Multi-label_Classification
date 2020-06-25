import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

history_path = '/home/jinHM/liziyi/Protein/Torch_Train/history/resnet50_0624_2_history.pt'
history = np.array(torch.load(history_path))

[train_loss, valid_loss,
 train_auc, valid_auc,
 train_macro_f1, valid_macro_f1,
 train_micro_f1, valid_micro_f1] = history.transpose()

# global sess
# with tf.Session() as session:  # 启动会话
#     sess = session
#     writer = tf.summary.FileWriter('/home/jinHM/liziyi/Protein/Torch_Train/history', sess.graph)
#     for i in range(history.shape[0]):
#         summary = tf.Summary()
#         summary.value.add(tag='train_micro_f1', simple_value=train_micro_f1[i])
#         summary.value.add(tag='train_macro_f1', simple_value=train_macro_f1[i])
#         summary.value.add(tag='train_roc_auc', simple_value=train_auc[i])
#         summary.value.add(tag='train_loss', simple_value=train_loss[i])
#
#         summary.value.add(tag='valid_micro_f1', simple_value=valid_micro_f1[i])
#         summary.value.add(tag='valid_macro_f1', simple_value=valid_macro_f1[i])
#         summary.value.add(tag='valid_roc_auc', simple_value=valid_auc[i])
#         summary.value.add(tag='valid_loss', simple_value=valid_loss[i])
#
#         writer.add_summary(summary, i)


plot = '/home/jinHM/liziyi/Protein/Torch_Train/resnet50_0624_2_history.png'
matplotlib.use("Agg")
plt.style.use("ggplot")
plt.figure()
plt.figure(figsize=(8, 9))
N = history.shape[0]
plt.plot(np.arange(0, N), train_loss, label="train_loss")
plt.plot(np.arange(0, N), valid_loss, label="val_loss")
plt.plot(np.arange(0, N), train_auc, label="train_auc")
plt.plot(np.arange(0, N), valid_auc, label="val_auc")
plt.plot(np.arange(0, N), train_micro_f1, label="train_micro_f1")
plt.plot(np.arange(0, N), valid_micro_f1, label="valid_micro_f1")

plt.title("")
plt.xlabel("Epoch #")
plt.ylabel("Loss/AUC/F1")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(plot)
