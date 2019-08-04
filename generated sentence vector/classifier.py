import torch
import torch.nn as nn
import torch.optim as optim
import time
from datautil.dataloader import batch_iter
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Classifier(object):
    def __init__(self, model=None, args=None, vocab=None, char_vocab=None):
        super(Classifier, self).__init__()  # 对继承自父类的属性进行初始化
        assert isinstance(model, nn.Module)
        self._model = model
        self._args = args
        self._vocab = vocab
        self._char_vocab = char_vocab

    def summary(self):
        print(self._model)

    def _draw(self, train_loss, val_loss):
        assert len(train_loss) == len(val_loss)
        epochs = len(train_loss)
        plt.subplot(211)
        plt.plot(range(epochs), train_loss, c='b')
        plt.subplot(212)
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        plt.plot(range(epochs), val_loss, c='r', linestyle='--')
        plt.xlabel('epoch')
        plt.ylabel('validate loss')
        plt.tight_layout()
        plt.show()

    def train(self, train_data, dev_data):
        # 优化器：更新模型参数
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                               lr=self._args.learning_rate,  # 1e-2
                               weight_decay=self._args.weight_decay)  # 0

        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._model.parameters()),
        #                       lr=self._args.learning_rate,  # 1e-2
        #                       momentum=0.9,
        #                       weight_decay=self._args.weight_decay,  # 1e-5
        #                       nesterov=True)

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 0.95**ep)

        # 迭代更新
        for i in range(self._args.epochs):
            self._model.train()

            start = time.time()
            train_loss = 0

            lr_scheduler.step()
            for batch_data in batch_iter(train_data, self._args.batch_size, self._vocab, device=self._args.device):
                self._model.zero_grad()
                pred, tgt = self._model(batch_data.wd_src)
                loss = self._calc_loss(pred, tgt)
                train_loss += loss.data.item()
                loss.backward()
                # gradient exploding
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, (self._model.parameters())), max_norm=1.0)

                optimizer.step()

            dev_loss = self._validate(dev_data)
            end = time.time()

            print('learning rate:', lr_scheduler.get_lr())
            print('[Epoch %d] train loss: %.3f  dev loss: %.3f' % (i, train_loss, dev_loss))
            print('time cost: %.3f' % (end-start))

    def _validate(self, dev_data):
        self._model.eval()

        dev_loss = 0
        cos_sim_lst = []
        with torch.no_grad():  # 确保在代码执行期间没有计算和存储梯度, 起到预测加速作用
            for batch_data in batch_iter(dev_data, self._args.batch_size, self._vocab, device=self._args.device):
                pred, tgt = self._model(batch_data.wd_src)
                loss = self._calc_loss(pred, tgt)
                dev_loss += loss.data.item()
                _, pred_enc = self._model.encoder(pred, batch_data.non_pad_mask)
                _, tgt_enc = self._model.encoder(tgt, batch_data.non_pad_mask)
                cos_sim_lst.append(self._cosine_sim(pred_enc[0][-1], tgt_enc[0][-1]))
            print('cosine similarity:', sum(cos_sim_lst) / len(cos_sim_lst))

        return dev_loss

    def evaluate(self, test_data):
        self._model.eval()
        cos_sim_lst = []
        test_loss = 0
        with torch.no_grad():  # 确保在代码执行期间没有计算和存储梯度, 起到预测加速作用
            for batch_data in batch_iter(test_data, self._args.batch_size, self._vocab, device=self._args.device):
                pred, tgt = self._model(batch_data.wd_src)
                loss = self._calc_loss(pred, tgt)
                test_loss += loss.data.item()
                _, pred_enc = self._model.encoder(pred, batch_data.non_pad_mask)
                _, tgt_enc = self._model.encoder(tgt, batch_data.non_pad_mask)
                cos_sim_lst.append(self._cosine_sim(pred_enc[0][-1], tgt_enc[0][-1]))

            print('cosine similarity:', sum(cos_sim_lst) / len(cos_sim_lst))

        print('===== test loss: %.3f =====' % test_loss)
        return test_loss

    def _calc_acc(self, pred, target):
        return torch.eq(torch.argmax(pred, dim=1), target).sum().item()

    def _calc_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def _cosine_sim(self, pred, tgt):  # 输入的维度数为2
        return F.cosine_similarity(pred, tgt).mean().item()  # 默认dim=1
