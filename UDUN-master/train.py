#!/usr/bin/python3
# coding=utf-8
import datetime
import argparse
import sys
sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from model.UDUNet import UDUN

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--savepath', default="../saveWeight", type=str)
    parser.add_argument('--datapath', default="../DIS5K/DIS-TR", type=str)
    parser.parse_args()
    return parser.parse_args()

def train(Dataset, Network):
    ## dataset
    args = parser()
    cfg = Dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=0.05, momen=0.9, decay=5e-4, epoch=48)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=4)

    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.epoch):

        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask, trunk, struct) in enumerate(loader):

            image, mask, trunk, struct = image.cuda(), mask.cuda(), trunk.cuda(), struct.cuda()
            out_trunk, out_struct, out_mask= net(image)

            trunk = F.interpolate(trunk, size=out_trunk.size()[2:], mode='bilinear')
            loss_t = F.binary_cross_entropy_with_logits(out_trunk, trunk)
            mask = F.interpolate(mask, size=out_mask.size()[2:], mode='bilinear')
            lossmask = F.binary_cross_entropy_with_logits(out_mask, mask) + iou_loss(out_mask, mask)
            struct = F.interpolate(struct, size=out_struct.size()[2:], mode='bilinear')
            loss_s = F.binary_cross_entropy_with_logits(out_struct, struct)
            loss = (loss_t + loss_s + lossmask) / 2

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_t': loss_t.item(), 'loss_s': loss_s.item(), 'lossmask': lossmask.item()}, global_step=global_step)
            if step % 75 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | lossmask=%.6f|' % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],lossmask.item()))

        if epoch > cfg.epoch * 3 / 4:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))

if __name__ == '__main__':
    train(dataset, UDUN)
