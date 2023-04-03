'''模型训练相关的函数'''

import numpy as np
import pickle
import cv2
import torch
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
from torchvision.datasets import ImageFolder
import sys
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import datetime

import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F


def calc_accuracy(model, loader, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
            if isinstance(outputs_batch, tuple):
                outputs_batch = model(inputs)[0]
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1
        try:

            APCER = living_wrong / (living_wrong + living_right)
            NPCER = spoofing_wrong / (spoofing_wrong + spoofing_right)
            ACER = (APCER + NPCER) / 2

            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            ACER = float("%.6f" % ACER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(e)
            print(living_right, living_wrong, spoofing_right, spoofing_wrong)
            return [accuracy, 0, 0, 0]

        print(living_right, living_wrong, spoofing_right, spoofing_wrong)
        return [accuracy, APCER, NPCER, ACER]
    else:
        return [accuracy]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, args, optimizer, multiplier, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = args.total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_last_lr()
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def train_base(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''

    print(args)

    # Initialize and open timer
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=0)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int32(args.train_epoch * 1 / 6),
                                                                              np.int32(args.train_epoch * 2 / 6),
                                                                              np.int32(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            if epoch == 0:
                continue

            batch_num += 1
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()

            output = model(data)
            if isinstance(output, tuple):
                output = output[0]

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # testing
        result_test = calc_accuracy(model, loader=test_loader, hter=True, verbose=True)
        print(result_test)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path, )


        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 100:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def test_base(model, cost, test_loader):
    '''
    :param model: 要测试的带参数模型
    :param cost: 损失函数
    :param test_loader: 测试集
    :return:
    '''

    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    true_wrong = 0
    true_right = 0
    false_wrong = 0
    false_right = 0

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)  # 数据是按照batch的形式喂入的,然后是这里的输出是全连接层的输出结果
        test_loss += (
            cost(output, target)).item()  # sum up batch 求loss 把所有loss值进行累加.一般分类器会有softmax,但是这里没有是因为集成在这个损失函数之中了
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability #输出的结果虽然会过sofamax转成概率值,但是相对打大小是不变的,所以直接输出取最大值对应的索引就是分类结果
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        compare_result = pred.eq(target.data.view_as(pred)).cpu()  # 判断输出和目标数据是不是一样,然后将比较结果转到cpu上
        # target=target.numpy()
        target = target.cpu()
        compare_result = np.array(compare_result)
        for i in range(len(compare_result)):
            if compare_result[i]:
                if target[i] == 1:
                    true_right += 1
                else:
                    false_right += 1
            else:
                if target[i] == 1:
                    true_wrong += 1
                else:
                    false_wrong += 1

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def deploy_base(model, img, transform):
    '''

    :param model: 模型
    :param img: PIL.Image的对象
    :transform 对应图像处理的操作,和test的一样
    :return:
    '''

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        img_tensor = img_tensor.cuda()
    result = model(img_tensor)

    if use_cuda:
        result = result.cpu()
    result = result.detach().numpy()
    result = result[0]

    return result


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(data.shape[0]):
        if label[i]==0:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color = "r",
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color = "b",
                     fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def sne_analysis(model, data_loader):
    '''
    用来可视化模型输出的特征向量,看看类内和类间的间距
    :param model: 输特征向量的模型,而不是最后分类结果的向量,一般是全连接层的倒数第二层的输出
    :param data_loader: 用来测试的数据的dataloader.推荐batch_size 不要太大.4或者8 较为合适
    :return:
    '''

    # for batch_idx, batch_sample in enumerate(data_loader):
    #     img_rgb, img_ir, img_depth, target = batch_sample['image_x'], batch_sample['image_ir'], \
    #                                          batch_sample['image_depth'], batch_sample[
    #                                              'binary_label']



    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # 求特征
            # feature = model(img_rgb,img_ir,img_depth)  # 把数据输入网络并得到输出，即进行前向传播
            # feature=feature[0]
            feature=model(data)
            feature = torch.flatten(feature, 1)

        if batch_idx == 0:
            feature_arr = np.array(feature)
            target_arr = np.array(target)
        else:
            feature_arr = np.vstack((feature_arr, np.array(feature)))
            target_arr = np.hstack((target_arr, np.array(target)))

        if batch_idx > 100:
            break

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = datetime.datetime.now()
    result = tsne.fit_transform(feature_arr)
    fig = plot_embedding(result, target_arr,
                         't-SNE embedding of the digits (time %.2fs)'
                         % ((datetime.datetime.now() - t0).total_seconds()))
    # plt.show()
    # plt.savefig
    plt.savefig('/home/shicaiwei/data/rgb_original.pdf', dpi=600, format='pdf')

