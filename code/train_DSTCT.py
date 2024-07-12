import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.distributions import Categorical
from tqdm import tqdm

from config import get_config
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.unet import UNet
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, ramps
from val_2D import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/FHPS', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='FHPS/DSTCT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000
                    , help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--b', type=float,
                    default=1.0, help='b')
parser.add_argument('--cps', type=float,
                    default=0.5, help='cps')
parser.add_argument('--y', type=float,
                    default=3.0, help='c')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "FHPS" in dataset:
        ref_dict = {"3": 179, "7": 357, "9":464,
                    "14": 714,"17": 892, "21": 1071, "28": 1428, "35": 1785, "140": 3570}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
 

def get_current_consistency_weight(epoch, weight):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    '''
    如果 Student 对特定样本的预测有偏差, EMA Teacher 有可能积累错误并强制学生跟随, 
    使错误分类不可逆转. 大多数方法对一致性约束应用了 ramp-up/down 操作来减轻偏差, 但不足以解决问题
    '''
    return weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# 锐化函数定义
def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def discrepancy_calc(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    assert v1.dim() == 4
    assert v2.dim() == 4
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)
    dis = torch.sum(mul) - torch.sum(inner)
    # dis = torch.sum(inner) - torch.sum(mul)
    dis = dis / (h * w)
    return dis

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # define models

    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    ema_model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()

    ema_model.load_from(config)
    for param in ema_model.parameters():
        param.detach_()

    model1 = UNet(in_chns=1, class_num=num_classes).cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    # SGD optimizer
    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    consistency_criterion = losses.mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    best_performance3 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            """
            最终得到的张量就是一个大小和类型与 "unlabeled_volume_batch" 相同的张量，其中每个元素都是从均值为 0，
            方差为 0.01 的正态分布中随机采样而来，并且每个元素满足 [-0.2, 0.2] 的范围限制，这个张量被赋予了变量名 "noise"。
            通常这种技巧被称为在张量中加入噪声。
            """
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)
                
            num_outputs1 = len(outputs1)
            num_outputs2 = len(outputs2)
            y_ori1 = torch.zeros((num_outputs1,) + outputs1[0].shape)
            y_ori2 = torch.zeros((num_outputs2,) + outputs2[0].shape)
            y_pseudo_label1 = torch.zeros((num_outputs1,) + outputs1[0].shape)
            y_pseudo_label2 = torch.zeros((num_outputs1,) + outputs1[0].shape)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            # cross pseudo losses
            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            
             
            for idx in range(num_outputs1):
                y_all = outputs1[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori1[idx] = y_prob_all
                y_pseudo_label1[idx] = sharpening(y_prob_all) 
            for idx in range(num_outputs2):
                y_all = outputs2[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori2[idx] = y_prob_all
                y_pseudo_label2[idx] = sharpening(y_prob_all) 
            
            loss_consist1 = 0
            for i in range(num_outputs1):
                for j in range(num_outputs1):
                    if i != j:
                        loss_consist1 += consistency_criterion(y_ori1[i], y_pseudo_label2[j])
            
            loss_consist2 = 0
            for i in range(num_outputs2):
                for j in range(num_outputs2):
                    if i != j:
                        loss_consist2 += consistency_criterion(y_ori2[i], y_pseudo_label1[j])
                        
           
            consistency_weight_cps = get_current_consistency_weight(iter_num // 150, args.cps)
            consistency_weight_mt = get_current_consistency_weight(iter_num // 150, args.consistency)
            b_weight = get_current_consistency_weight(iter_num // 150, args.b)
            y_weight = get_current_consistency_weight(iter_num // 150, args.y)
            
            target_pred1 = F.softmax(outputs1)
            target_pred2 = F.softmax(outputs2)
            l_cr = discrepancy_calc(target_pred1, target_pred2)
            
            
            if iter_num < 1000:
                consistency_loss2_2 = 0.0
                consistency_loss1_1 = 0.0
            else:
                consistency_loss1_1 = torch.mean((outputs_soft1[args.labeled_bs:] - ema_output_soft) ** 2)
                consistency_loss2_2 = torch.mean((outputs_soft2[args.labeled_bs:] - ema_output_soft) ** 2)

            # adjust ratio of loss here
            model1_loss = loss1 + consistency_weight_cps * pseudo_supervision1 + b_weight * loss_consist1 + consistency_weight_mt * consistency_loss1_1
            model2_loss = loss2 + consistency_weight_cps * pseudo_supervision2 + b_weight * loss_consist2 + consistency_weight_mt * consistency_loss2_2
            
            loss = model1_loss + model2_loss + l_cr * y_weight
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            # update_ema_variables(model2, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            iter_num = iter_num + 1

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', consistency_weight_cps, iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss, iter_num)
            logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction', outputs[1, ...] * 50, iter_num)

                outputs = torch.argmax(torch.softmax(outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction', outputs[1, ...] * 50, iter_num)

                # outputs = torch.argmax(torch.softmax(ema_output, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/ema_model_Prediction', outputs[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                # model1 evaluation
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model1,
                        classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance1, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                # model2 evaluation
                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(
                        best_performance2, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

                # ema model evaluation
                # ema_model.eval()
                # metric_list = 0.0
                # for i_batch, sampled_batch in enumerate(valloader):
                #     metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], ema_model,
                #                                   classes=num_classes, patch_size=args.patch_size)
                #     metric_list += np.array(metric_i)
                # metric_list = metric_list / len(db_val)
                # for class_i in range(num_classes - 1):
                #     writer.add_scalar('info/ema_model_val_{}_dice'.format(class_i + 1), metric_list[class_i, 0],
                #                       iter_num)
                #     writer.add_scalar('info/ema_model_val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1],
                #                       iter_num)

                # performance3 = np.mean(metric_list, axis=0)[0]

                # mean_hd953 = np.mean(metric_list, axis=0)[1]
                # writer.add_scalar('info/ema_model_val_mean_dice', performance3, iter_num)
                # writer.add_scalar('info/ema_model_val_mean_hd95', mean_hd953, iter_num)

                # if performance3 > best_performance3:
                #     best_performance3 = performance3
                #     save_mode_path = os.path.join(snapshot_path, 'ema_model_iter_{}_dice_{}.pth'.format(iter_num, round(
                #         best_performance3, 4)))
                #     save_best = os.path.join(snapshot_path, '{}_best_ema_model.pth'.format(args.model))
                #     torch.save(ema_model.state_dict(), save_mode_path)
                #     torch.save(ema_model.state_dict(), save_best)

                # logging.info(
                #     'iteration %d : ema_model_mean_dice : %f ema_model_mean_hd95 : %f' % (
                #         iter_num, performance3, mean_hd953))

            if iter_num % 3000 == 0:
            # if iter_num % 200 == 0:
                save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

                # save_mode_path = os.path.join(snapshot_path, 'ema_model_iter_' + str(iter_num) + '.pth')
                # torch.save(ema_model.state_dict(), save_mode_path)
                # logging.info("save ema_model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
    