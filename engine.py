from typing import Iterable
import json

import numpy as np
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import sklearn, sklearn.metrics
import pandas as pd
import sklearn.metrics

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_scheduler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        images = samples["img"].float().to(device)
        labels = samples["lab"].to(device)

        loss = model(images=images, labels=labels)
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(net.parameters(), max_norm=10., norm_type=2)	# For cosine loss
        optimizer.step()
        lr_scheduler.step_update(epoch * len(data_loader) + data_iter_step)

        loss_value = loss.item()

        # todo: improve here
        torch.cuda.synchronize()

        metric_logger.update(closs=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    running_total = 0.
    running_clean_correct = 0
    sick_sick = 0
    sick_total = 0
    health_health = 0
    health_total = 0
    all_loss = list()
    task_outputs = {}
    task_targets = {}
    scores = []
    labs = []
    outs=[]
    for task in range(2):
        task_outputs[task] = []
        task_targets[task] = []
    for batch_idx, samples in enumerate(data_loader):
        images = samples["img"].to(device)
        labels = samples["lab"].to(device)
        imgname = samples["img_name"][0]
        label = []
        label.append('Sick' in imgname)
        label.append('Health' in imgname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        labs.append(label)
        with torch.no_grad():
            pred, loss = model(images=images, labels=labels, return_preds=True)
        out = pred.cpu().detach().numpy()[0]
        outargmax = []
        outargmax.append(np.argmax(out) == 0)
        outargmax.append(np.argmax(out) == 1)
        outargmax = np.asarray(outargmax).T
        outs.append(outargmax)
        score = F.softmax(pred.float().cpu().detach(), dim=-1).numpy()[0]
        scores.append(score)
        loss_item = loss.item()
        all_loss.append(loss_item)
        aa, clean_preds = pred.max(1)
        bb, target = labels.max(1)
        running_clean_correct += clean_preds.eq(target).sum().item()
        running_total += images.size(0)
        if target.detach().cpu().numpy() == 0:
            sick_total += 1
            if clean_preds.detach().cpu().numpy() == 0:
                sick_sick += 1
        else:
            health_total += 1
            if clean_preds.detach().cpu().numpy() == 1:
                health_health += 1

    all_loss_np = np.asarray(all_loss)
    task_accs = np.asarray(running_clean_correct / running_total)
    Sensitivity = np.asarray(sick_sick / sick_total)
    Specificity = np.asarray(health_health / health_total)
    Specificity = np.mean(Specificity[~np.isnan(Specificity)])
    Sensitivity = np.mean(Sensitivity[~np.isnan(Sensitivity)])
    avgauc = []
    avgf1 = []
    avgrecall = []
    avgprecision = []
    for i in range(2):
        if len(np.unique(np.asarray(labs)[:, i])) > 1:
            labels = np.asarray(labs)[:, i].astype(bool)
            scoresi = np.asarray(scores)[:, i]
            preds = np.asarray(outs)[:, i]
            auc = sklearn.metrics.roc_auc_score(labels, scoresi)
            avgauc.append(auc)
            f1 = sklearn.metrics.f1_score(labels, preds)
            # # 计算召回率
            recall = sklearn.metrics.recall_score(labels, preds)
            precision = sklearn.metrics.precision_score(labels, preds)
            avgf1.append(f1)
            avgrecall.append(recall)
            avgprecision.append(precision)

    task_aucs = np.asarray(avgauc)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    task_f1 = np.asarray(avgf1)
    f1 = np.mean(task_f1[~np.isnan(task_f1)])
    task_recall = np.asarray(avgrecall)
    recall = np.mean(task_recall[~np.isnan(task_recall)])
    task_precision = np.asarray(avgprecision)
    precision = np.mean(task_precision[~np.isnan(task_precision)])

    return all_loss_np, task_accs, Sensitivity, Specificity, auc, f1, recall, precision


def infer_one_epoch(model: torch.nn.Module, data_loader: Iterable, device: torch.device, res_file):
    model.eval()

    prefix_img = torch.tensor(data_loader.dataset.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
    prefix_img = prefix_img.to(device)

    softmax_func = torch.nn.Softmax(dim=1)

    results = dict()
    for examples, labels, example_mask, images, image_names in data_loader:
        examples = examples.to(device)
        labels = labels.to(device)
        images = images.to(device)

        with torch.no_grad():
            preds = model(examples, labels, images=images, prefix_img=prefix_img, return_preds=True)
        scores = softmax_func(preds)
        scores_np = scores.data.cpu().numpy().astype(float)

        for i, name in enumerate(image_names):
            results[name[:-4]] = (scores_np[i, 0], scores_np[i, 1])
    # save results in json
    with open(res_file, 'w') as f:
        json.dump(results, f)

    return
