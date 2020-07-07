import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


class DrawHistory:
    def __init__(self, history_path):
        self.history_path = history_path
        self.history = np.array(torch.load(history_path))

        [self.train_loss, self.valid_loss,
         self.train_auc, self.valid_auc,
         self.train_macro_f1, self.valid_macro_f1,
         self.train_micro_f1, self.valid_micro_f1] = self.history.transpose()

    def using_tensorboard(self, tb_path):
        global sess
        with tf.Session() as session:  # 启动会话
            sess = session
            writer = tf.summary.FileWriter(tb_path, sess.graph)
            for i in range(history.shape[0]):
                summary = tf.Summary()
                summary.value.add(tag='train_micro_f1', simple_value=self.train_micro_f1[i])
                summary.value.add(tag='train_macro_f1', simple_value=self.train_macro_f1[i])
                summary.value.add(tag='train_roc_auc', simple_value=self.train_auc[i])
                summary.value.add(tag='train_loss', simple_value=self.train_loss[i])

                summary.value.add(tag='valid_micro_f1', simple_value=self.valid_micro_f1[i])
                summary.value.add(tag='valid_macro_f1', simple_value=self.valid_macro_f1[i])
                summary.value.add(tag='valid_roc_auc', simple_value=self.valid_auc[i])
                summary.value.add(tag='valid_loss', simple_value=self.valid_loss[i])

                writer.add_summary(summary, i)

    def using_matplotlib(self, plot_path):
        plot = plot_path
        matplotlib.use("Agg")
        plt.style.use("ggplot")
        plt.figure()
        plt.figure(figsize=(8, 9))
        N = history.shape[0]
        plt.plot(np.arange(0, N), self.train_loss, label="train_loss")
        plt.plot(np.arange(0, N), self.valid_loss, label="val_loss")
        plt.plot(np.arange(0, N), self.train_auc, label="train_auc")
        plt.plot(np.arange(0, N), self.valid_auc, label="val_auc")
        plt.plot(np.arange(0, N), self.train_micro_f1, label="train_micro_f1")
        plt.plot(np.arange(0, N), self.valid_micro_f1, label="valid_micro_f1")

        plt.title("")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/AUC/F1")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.savefig(plot)

    def save(self):
        with open('/home/jinHM/liziyi/Protein/Torch_Train/save.csv', 'w') as f:
            f.write('metrics,')
            f.write(','.join([str(x + 1) for x in range(30)]))
            f.write('\n')

            f.write('train_loss,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.train_loss]))
            f.write('\n')

            f.write('valid_loss,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.valid_loss]))
            f.write('\n')

            f.write('train_auc,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.train_auc]))
            f.write('\n')

            f.write('valid_auc,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.valid_auc]))
            f.write('\n')

            f.write('train_macro_f1,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.train_macro_f1]))
            f.write('\n')

            f.write('valid_macro_f1,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.valid_macro_f1]))
            f.write('\n')

            f.write('train_micro_f1,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.train_micro_f1]))
            f.write('\n')

            f.write('valid_micro_f1,')
            f.write(','.join(['{:.4f}'.format(x) for x in self.valid_micro_f1]))
            f.write('\n')


if __name__ == '__main__':
    history_path = '/home/jinHM/liziyi/Protein/Torch_Train/history/resnet50_0624_2_history.pt'
    draw = DrawHistory(history_path)
    # draw.using_matplotlib('/home/jinHM/liziyi/Protein/Torch_Train/resnet50_0624_2_history.png')
    # draw.using_tensorboard('/home/jinHM/liziyi/Protein/Torch_Train/history')
    draw.save()
