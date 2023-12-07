import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.set_num_threads(1)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import json
import time
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory
from lr_scheduler import build_scheduler
import util.misc as misc
from engine import train_one_epoch, val_one_epoch
from util.datasets_pneum import Shanxi_Dataset
from phe_llm.mm_adaptation import phellm
import random
def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True
    print(cudnn.benchmark, cudnn.deterministic, cudnn.enabled)


def get_args_parser(exp_name):
    parser = argparse.ArgumentParser('Pneumollm Description', add_help=False)
    # NOTICE: parameters that should be adjust before running

    parser.add_argument('--num_epochs', default=100, type=int)
    # NOTICE: adjust max_seq_len with the variation of image feature
    parser.add_argument('--max_seq_len', type=int, default=64, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--output_dir', default='./outputs/'+exp_name+'_weights', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs/'+exp_name+'_logs', help='path where to tensorboard log')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--data_root', type=str, default='/dataset/', help='The path of dataset')
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
    parser.add_argument('--min_lr', type=float, default=1e-7, metavar='LR',
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
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    return parser


def main(args):
    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # set_seed
    setup_seed(args.seed)
    dataset_train = Shanxi_Dataset(imgpath=args.data_root+'segimages_224_resize',
                 txtpath=args.data_root+'train.txt')
    dataset_val = Shanxi_Dataset(imgpath=args.data_root+'segimages_224_resize',
                 txtpath=args.data_root+'val.txt')
    if os.path.exists(args.output_dir):
        print('output_dir exist!')
    else:
        os.mkdir(args.output_dir)
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    # define the model
    model = phellm(args)
    checkpoint_file = '/checkpoint.pth'
    checkpoint = torch.load(checkpoint_file, map_location="cpu")['model']
    model_dict = model.state_dict()

    pretrained_dict = {key: value for key, value in checkpoint.items() if
                       (key in model_dict)}

    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    print("base lr: %.2e" % (args.learning_rate * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.learning_rate)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))

    # mixed precision scaler
    lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))
    best_val_acc = 0.
    best_val_epoch=0
    best_val_avg=0
    best_val_auc = 0
    best_val_speci = 0
    best_val_sens = 0
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, lr_scheduler=lr_scheduler)

    print(f"Start training for {args.num_epochs} epochs")
    start_time = time.time()
    model.train(True)

    for epoch in range(args.start_epoch, args.num_epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, lr_scheduler,
            log_writer=log_writer,
            args=args
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }
        # validation
        data_loader_val.sampler.set_epoch(epoch)
        loss_val, acc, Sensitivity, Specificity, auc = val_one_epoch(model, data_loader_val, device)
        print('Epoch', epoch, '  Val mean loss: ', loss_val.mean())
        task_avg = (acc + Sensitivity + Specificity + auc) / 4
        print('Epoch', epoch, '  Val acc loss: ', task_avg.mean())
        if task_avg > best_val_avg:
            best_val_avg = task_avg
            best_val_epoch=epoch
            best_val_speci=Specificity
            best_val_sens=Sensitivity
            best_val_acc=acc
            best_val_auc=auc
            model_save_path2 = args.output_dir + '/trainable_bestvalpt.pth'
            misc.save_model(
                args=args,epoch=best_val_epoch,model_save_path=model_save_path2, model_state_dict=model.state_dict(), optimizer=optimizer,
                lr_scheduler= lr_scheduler)
        print(" Best Epoch: ", best_val_epoch, "!","New best val auc:", "%.4f" % best_val_auc,"acc:", "%.4f" % best_val_acc,"spe:", "%.4f" % best_val_speci,"sen:", "%.4f" % best_val_sens,"avg:", "%.4f" % best_val_avg,)
        log_writer.add_scalar("val_loss", loss_val.mean(), epoch)
        log_writer.add_scalar("val_avg", task_avg.mean(), epoch)
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.write('Epoch' + str(epoch) + '  val mean loss: ' + "%.4f" % loss_val.mean() + "\n")
                f.write('Epoch' + str(epoch) + '  val average: ' + "%.4f" % task_avg.mean() + "\n")
                f.write("Best val average:" + "%.4f" % best_val_avg + "! Best Epoch: " + str(
                    best_val_epoch) + "!\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    import torch
    exp_name = 'pneumollm'
    args = get_args_parser(exp_name)
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
