import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
torch.set_num_threads(1)
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import sklearn, sklearn.metrics
import pandas as pd
import sklearn.metrics

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import util.misc as misc
from util.datasets_pneum import Shanxi_Dataset
from phe_llm.mm_adaptation import phellm


def get_args_parser(exp_name):
    parser = argparse.ArgumentParser('Pneumollm Description', add_help=False)
    # NOTICE: parameters that should be adjust before running
    parser.add_argument('--num_epochs', default=100, type=int)
    # NOTICE: adjust max_seq_len with the variation of image feature
    parser.add_argument('--max_seq_len', type=int, default=64, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--output_dir', default='./outputs/'+exp_name+'_weights', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs/'+exp_name+'_logs', help='path where to tensorboard log')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--data_root', type=str, default='/dataset/')

    parser.add_argument('--multiscale', type=int, default=4)
    parser.add_argument('--RPO_K', type=int, default=4)
    # todo: remove these parameters
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # If GPU memory is not sufficient, adjust these parameters
    parser.add_argument('--bits', default='16bit', type=str, choices=['4bit', '8bit', '16bit'],
                        help='Quantization bits for training, model_save_path2 by default')
    parser.add_argument('--gradient_checkpointing', default=False,
                        help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--pin_mem', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # These parameter do not need adjust
    parser.add_argument('--llama_model_path', default='.../LLaMA-7B', type=str, help='path of llama model')
    parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL', help='Name of llm model to train')
    # todo: remove distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # block is not supported now
    parser.add_argument('--adapter_type', type=str, default='attn', metavar='LENGTH', choices=['block', 'attn'], help='the insert position  of adapter layer')
    parser.add_argument('--visual_adapter_type', type=str, default='router', metavar='LENGTH', choices=['normal', 'router', 'router_block'], help='the type of adapter layer')
    parser.add_argument('--adapter_dim', type=int, default=8, metavar='LENGTH', help='the dims of adapter layer')
    parser.add_argument('--hidden_proj', type=int, default=128, metavar='LENGTH', help='the visual adapter dim')
    parser.add_argument('--temperature', type=float, default=10., metavar='LENGTH', help='the temperature of router')
    parser.add_argument('--adapter_scale', type=float, default=1., metavar='LENGTH', help='the scales of adapter layer')
    parser.add_argument('--drop_path', type=float, default=0., metavar='LENGTH', help='drop path')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay (default: 0.05)')
    parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=float, default=2, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N', help='epochs to decay LR')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='gamma')
    parser.add_argument('--warmup_prefix', default=True)
    parser.add_argument('--lr_scheduler_name', type=str, default='cosine')
    parser.add_argument('--multisteps', default=[])
    # Dataset parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    return parser


def main(args):
    device = torch.device(args.device)

    dataset_val = Shanxi_Dataset(imgpath=args.data_root + 'segimages_224_resize',
                                 txtpath=args.data_root + 'val.txt')

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=False)

    print("Sampler_val = %s" % str(sampler_val))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # define the model
    model = phellm(args)

    # checkpoint_file=args.output_dir + args.experi+'/'+'trainable_bestvalpt.pth'
    checkpoint_file = '/checkpoint.pth'

    checkpoint = torch.load(checkpoint_file, map_location="cpu")['model']
    model_dict = model.state_dict()

    pretrained_dict = {key: value for key, value in checkpoint.items() if
                       (key in model_dict)}

    model.load_state_dict(pretrained_dict, strict=False)


    model.to(device)

    print('load ', checkpoint_file, ' weights')
    model.eval()
    sick_sick = 0
    sick_total = 0
    health_health = 0
    health_total = 0
    outs = []
    scores=[]
    labs = []
    data_results = []
    running_clean_correct=0
    running_total=0
    loss = torch.zeros(1).to(device).double()
    for data_iter_step, sample in enumerate(data_loader_val):
        images = sample["img"].to(device)
        labels = sample["lab"].to(device)
        imgname=sample["img_name"][0]
        labelname = imgname.split('_')[0]
        label = []
        label.append('Sick' in imgname)
        label.append('Health' in imgname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        labs.append(label)
        with torch.no_grad():
            preds, loss_test = model(images=images, labels=labels,return_preds=True)

        loss+=loss_test
        out = preds.cpu().detach().numpy()[0]
        aa, clean_preds = preds.max(1)
        bb, target = labels.max(1)
        running_clean_correct += clean_preds.eq(target).sum().item()
        running_total += images.size(0)
        if np.argmax(label) == 0:
            sick_total += 1
            if np.argmax(out) == 0:
                sick_sick += 1
        else:
            health_total += 1
            if np.argmax(out) == 1:
                health_health += 1

        outargmax = []
        outargmax.append(np.argmax(out) == 0)
        outargmax.append(np.argmax(out) == 1)
        outargmax = np.asarray(outargmax).T
        outs.append(outargmax)
        score = F.softmax(preds.cpu().detach(), dim=-1).numpy()[0]
        scores.append(score)
        result = {}
        result["Image Index"] = imgname
        result["Finding Labels"] = labelname
        pred_labels = ""
        pred_score=""
        aa = np.argmax(out)
        if aa == 0:
            pred_labels = pred_labels + 'Sick'
        else:
            pred_labels = pred_labels + 'Health'
        result["Pred Labels"] = pred_labels
        result["Pred Sick_Score"] = pred_score+str(score[0])
        result["Pred Health_Score"] = pred_score+str(score[1])
        data_results.append(result)
    loss = loss.sum()
    print('test loss:', loss)
    task_accs = np.asarray(running_clean_correct / running_total)
    print('task_accs:', task_accs)
    df_data_results=pd.DataFrame(data_results)

    df_data_results.to_csv('./outputs/test_data_result.csv', index=False)
    results = []
    avgacc = []
    avgauc = []
    avgf1 = []
    avgrecall = []
    avgprecision = []
    for i in range(2):
        result = {}
        if i == 0:
            result["Pathology"] = "Sick"
        else:
            result["Pathology"] = "Health"
        if len(np.unique(np.asarray(labs)[:, i])) > 1:
            labels = np.asarray(labs)[:, i].astype(bool)
            preds = np.asarray(outs)[:, i]
            scoresi = np.asarray(scores)[:, i]
            auc = sklearn.metrics.roc_auc_score(labels, scoresi)
            acc = sklearn.metrics.accuracy_score(labels, preds)
            f1 = sklearn.metrics.f1_score(labels, preds)

            recall = sklearn.metrics.recall_score(labels, preds)
            precision = sklearn.metrics.precision_score(labels, preds)
            result["AUC"] = auc
            result["Acc"] = acc
            result["F1"] = f1
            result["Recall"] = recall
            result["Precision"] = precision
            avgauc.append(auc)
            avgacc.append(acc)
            avgf1.append(f1)
            avgrecall.append(recall)
            avgprecision.append(precision)
        results.append(result)
    result = {}
    result['Pathology'] = 'AVG'
    task_aucs = np.asarray(avgauc)
    result["AUC"] = np.mean(task_aucs[~np.isnan(task_aucs)])
    task_accs = np.asarray(avgacc)
    result["Acc"] = np.mean(task_accs[~np.isnan(task_accs)])
    task_f1 = np.asarray(avgf1)
    result["F1"] = np.mean(task_f1[~np.isnan(task_f1)])
    task_recall = np.asarray(avgrecall)
    result["Recall"] = np.mean(task_recall[~np.isnan(task_recall)])
    task_precision = np.asarray(avgprecision)
    result["Precision"] = np.mean(task_precision[~np.isnan(task_precision)])
    Sensitivity = np.asarray(sick_sick / sick_total)
    Specificity = np.asarray(health_health / health_total)
    Specificity = np.mean(Specificity[~np.isnan(Specificity)])
    Sensitivity = np.mean(Sensitivity[~np.isnan(Sensitivity)])
    result["Specificity"] = Specificity
    result["Sensitivity"] = Sensitivity
    results.append(result)
    df = pd.DataFrame(results)
    print(df)

    df.to_csv('./outputs/test_result.csv', index=False)



if __name__ == '__main__':
    exp_name = 'pneumollm'
    args = get_args_parser(exp_name)
    args = args.parse_args()
    main(args)



