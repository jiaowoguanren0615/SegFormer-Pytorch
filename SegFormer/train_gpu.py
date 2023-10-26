import os
import re
import torch
import datetime
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from timm.models import create_model
from models import *
from datasets import *
from utils.augmentations import get_train_augmentation, get_val_augmentation
from utils.losses import get_loss
from utils.schedulers import get_scheduler, create_lr_scheduler
from utils.optimizers import get_optimizer
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from engine import train_one_epoch, evaluate




def get_argparser():
    parser = argparse.ArgumentParser('Pytorch SegFormer Models training and evaluation script', add_help=False)

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./CityScapesDataset',help="path to Dataset")
    parser.add_argument("--image_size", type=int, default=[512, 512], help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")
    parser.add_argument("--pin_mem", type=bool, default=False, help="Dataloader ping_memory")
    parser.add_argument("--batch_size", type=int, default=4,help='batch size (default: 4)') # consume approximately 3G GPU-Memory
    parser.add_argument("--val_batch_size", type=int, default=2,help='batch size for validation (default: 2)')

    # SegFormer Options
    parser.add_argument("--model", type=str, default='make_SegFormerB1', help='model name')

    # Train Options
    parser.add_argument("--amp", type=bool, default=True, help='auto mixture precision') # There may be some problems when loading weights, such as: ComplexFloat
    parser.add_argument("--epochs", type=int, default=2, help='total training epochs')
    parser.add_argument("--device", type=str, default='cuda:0', help='device (cuda:0 or cpu)')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='num_workers, set it equal 0 when run programs in win platform')
    parser.add_argument("--DDP", type=bool, default=False)
    parser.add_argument("--train_print_freq", type=int, default=100)
    parser.add_argument("--val_print_freq", type=int, default=50)

    # Loss Options
    parser.add_argument("--loss_fn_name", type=str, default='OhemCrossEntropy')

    # Optimizer & LR-scheduler Options
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')

    parser.add_argument("--lr_scheduler", type=str, default='WarmupPolyLR')
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--lr_warmup", type=int, default=10)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.1)

    # save checkpoints
    parser.add_argument("--save_weights_dir", default='./save_weights', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default='./', help='SummaryWriter save dir')
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    return parser



def main(args):

    if not os.path.exists(args.save_weights_dir):
        os.makedirs(args.save_weights_dir)

    # start = time.time()
    best_mIoU = 0.0
    device = args.device

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_transform = get_train_augmentation(args.image_size, seg_fill=args.ignore_label)
    val_transform = get_val_augmentation(args.image_size)

    train_set = CityScapes(args.data_root, 'train', train_transform)
    valid_set = CityScapes(args.data_root, 'val', val_transform)

    model = make_SegFormerB1(num_classes=args.num_classes)

    if args.DDP:
        sampler = DistributedSampler(train_set, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[args.gpu_id])
    else:
        sampler = RandomSampler(train_set)


    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True, pin_memory=args.pin_mem, sampler=sampler)

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem)

    iters_per_epoch = len(train_set) // args.batch_size

    loss_fn = get_loss(args.loss_fn_name, train_set.ignore_label, None)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # scheduler = get_scheduler(args.lr_scheduler, optimizer, args.epochs * iters_per_epoch, args.lr_power,
    #                           iters_per_epoch * args.lr_warmup, args.lr_warmup_ratio)

    scheduler = create_lr_scheduler(optimizer, len(trainloader), args.epochs, warmup=True)

    scaler = GradScaler(enabled=args.amp) if torch.cuda.is_bf16_supported() else None

    # writer = SummaryWriter(str(args.save_dir / 'logs'))

    if args.resume:
        checkpoint_save_path = './save_weights/best_model.pth'
        checkpoint = torch.load(checkpoint_save_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_mIoU = checkpoint['best_mIoU']
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
                    
        print(f'The Best MeanIou is {best_mIoU:.4f}')


    model = model.to(device)
    for epoch in range(args.epochs):

        mean_loss, lr = train_one_epoch(args, model, optimizer, loss_fn, trainloader, sampler, scheduler,
                                     epoch, device, args.train_print_freq, scaler)

        confmat = evaluate(args, model, valloader, device, args.val_print_freq)

        val_info = str(confmat)
        print(val_info)


        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        with open(results_file, 'r') as file:
            text = file.read()
        match = re.search(r'mean IoU:\s+(\d+\.\d+)', text)
        if match:
            mean_iou = float(match.group(1))
            
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            checkpoint_save = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_mIoU": best_mIoU
            }
            if args.amp:
                checkpoint_save['scaler'] = scaler.state_dict()
            torch.save(checkpoint_save, f'{args.save_weights_dir}/best_model.pth')


    # writer.close()
    # end = time.gmtime(time.time() - start)

    # table = [
    #     ['Best mIoU', f"{best_mIoU:.2f}"],
    #     ['Total Training Time', time.strftime("%H:%M:%S", end)]
    # ]
    # print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Pytorch SegFormer Models training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()
    fix_seeds(2023)
    setup_cudnn()
    # gpu = setup_ddp()
    main(args)
    # cleanup_ddp()
