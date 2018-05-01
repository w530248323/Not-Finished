import os
import sys
import shutil
import time
import glob
import signal
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from data_loader_jpeg import VideoFolder
from generate_model import generate_model
from callbacks import PlotLearning, MonitorLRDecay, AverageMeter
from torch.autograd import Variable
from torchvision.transforms import *
from opts import parse_opts
# from model import ConvColumn
from visualdl import LogWriter
import torch.onnx

best_prec1 = 0
opt = parse_opts()
model_name = opt.model + str(opt.model_depth) \
             + '_' + opt.resnet_shortcut \
             + '_' + str(opt.sample_duration) \
             + '_' + str(opt.sample_height) \
             + '_' + str(opt.sample_width)

logdir = "./workspace/{}".format(model_name)
logger = LogWriter(logdir, sync_cycle=100)

# mark the components with 'train' label.
with logger.mode("train"):
    # create a scalar component called 'scalars/'
    scalar_train_loss = logger.scalar("scalars/scalar_train_loss")
    scalar_train_acc = logger.scalar("scalars/scalar_train_acc")
    scalar_val_loss = logger.scalar("scalars/scalar_val_loss")
    scalar_val_acc = logger.scalar("scalars/scalar_val_acc")


def main():
    global best_prec1
    # set run output folder
    print("=> Output folder for this run -- {}".format(model_name))
    save_dir = os.path.join(opt.output_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # adds a handler for Ctrl+C
    def signal_handler():
        """
        Remove the output dir, if you exit with Ctrl+C and
        if there are less then 3 files.
        It prevents the noise of experimental runs.
        """
        num_files = len(glob.glob(save_dir + "/*"))
        if num_files < 1:
            shutil.rmtree(save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # create model
    model = generate_model(opt)

    dummy_input = Variable(torch.randn(opt.batch_size, 3, opt.sample_duration, opt.sample_height, opt.sample_width)).cuda()
    torch.onnx.export(model, dummy_input, "{}.onnx".format(model_name))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.checkpoint):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.checkpoint)
            opt.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(
                opt.checkpoint))

    # find best cudnn configuration
    cudnn.benchmark = True

    transform = Compose([
        CenterCrop((96, 170)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_data = VideoFolder(root=opt.train_data_folder,
                             csv_file_input=opt.train_labels,
                             csv_file_labels=opt.labels,
                             clip_size=opt.sample_duration,
                             nclips=opt.nclips,
                             step_size=opt.step_size,
                             is_val=False,
                             transform=transform,
                             )

    print(" > Using {} processes for data loader.".format(
        opt.num_workers))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True,
        drop_last=True)

    val_data = VideoFolder(root=opt.val_data_folder,
                           csv_file_input=opt.val_labels,
                           csv_file_labels=opt.labels,
                           clip_size=opt.sample_duration,
                           nclips=opt.nclips,
                           step_size=opt.step_size,
                           is_val=True,
                           transform=transform,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True,
        drop_last=True)

    assert len(train_data.classes) == opt.n_classes

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # define optimizer
    lr = opt.lr
    last_lr = opt.last_lr
    momentum = opt.momentum
    weight_decay = opt.weight_decay
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    if opt.eval_only:
        validate(val_loader, model, criterion, train_data.classes_dict)
        return

    # set callbacks
    plotter = PlotLearning(os.path.join(
        save_dir, "plots"), opt.n_classes)
    lr_decayer = MonitorLRDecay(0.6, 3)
    val_loss = 9999999

    # set end condition by num epochs
    num_epochs = int(opt.n_epochs)
    if num_epochs == -1:
        num_epochs = 999999

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(num_epochs))
    start_epoch = opt.start_epoch if opt.resume else 0
    
    train_step = 0
    val_step = 0

    for epoch in range(start_epoch, num_epochs):
        lr = lr_decayer(val_loss, lr)
        print(" > Current LR : {}".format(lr))

        if lr < last_lr and last_lr > 0:
            print(" > Training is done by reaching the last learning rate {}".
                  format(last_lr))
            sys.exit(1)

        # train for one epoch
        start_time_epoch = time.time()
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch, train_step)
        print(" > Time taken for this 1 train epoch = {}".
              format(time.time() - start_time_epoch))

        # evaluate on validation set
        start_time_epoch = time.time()
        val_loss, val_top1, val_top5 = validate(val_loader, model, criterion, val_step)
        print(" > Time taken for this 1 validation epoch = {}".
              format(time.time() - start_time_epoch))

        # plot learning
        plotter_dict = {'loss': train_loss, 'val_loss': val_loss, 'acc': train_top1, 'val_acc': val_top1,
                        'learning_rate': lr}
        plotter.plot(plotter_dict)

        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_prec1
        best_prec1 = max(val_top1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': opt.model + str(opt.model_depth),
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, train_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input_vars = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        model.zero_grad()

        # compute output and loss
        output = model(input_vars)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use VisualDL to retrieve metrics
        # scalar
        scalar_train_loss.add_record(train_step, float(loss))
        scalar_train_acc.add_record(train_step, float(prec1))

        train_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, top1=top1,
                                                                  top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, val_step):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    class_to_idx=None

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    targets_list = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        input_vars = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        # compute output and loss
        output = model(input_vars)
        loss = criterion(output, target_var)

        if opt.eval_only:
            logits_matrix.append(output.cpu().data.numpy())
            targets_list.append(target.cpu().numpy())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        scalar_val_loss.add_record(val_step, float(loss))
        scalar_val_acc.add_record(val_step, float(prec1))
        
        val_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    if opt.eval_only:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        # youtube_ids_list = np.asarray(youtube_ids_list)
        # print(logits_matrix.shape, targets_list.shape, youtube_ids_list.shape)
        save_results(logits_matrix, targets_list, class_to_idx)
    return losses.avg, top1.avg, top5.avg


def save_results(logits_matrix, targets_list, class_to_idx):
    print("Saving inference results ...")
    path_to_save = os.path.join(
        opt.output_dir, model_name, "test_results.pkl")
    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list, class_to_idx], f)


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(
        opt.output_dir, model_name, filename)
    model_path = os.path.join(
        opt.output_dir, model_name, 'model_best.pth')
    torch.save(state, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, model_path)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.cpu().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
