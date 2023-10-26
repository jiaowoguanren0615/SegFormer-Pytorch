import torch
import math
from torch.nn import functional as F
from tqdm import tqdm
from utils.metrics import Metrics
from torch.cuda.amp import autocast
import utils.distributed_utils as utils



def train_one_epoch(args, model, optimizer, loss_fn, dataloader, sampler, scheduler,
                    epoch, device, print_freq, scaler=None):
    model.train()

    if args.DDP:
        sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for iter, (img, lbl) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        img = img.to(device)
        lbl = lbl.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(enabled=args.amp):
                logits = model(img)
                loss = loss_fn(logits, lbl)
        else:
            logits = model(img)
            loss = loss_fn(logits, lbl)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.cuda.synchronize()

        loss_value = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value, lr=lr)

    torch.cuda.empty_cache()

    return metric_logger.meters["loss"].global_avg, lr



@torch.no_grad()
def evaluate(args, model, dataloader, device, print_freq):
    model.eval()

    confmat = utils.ConfusionMatrix(args.num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, labels in metric_logger.log_every(dataloader, print_freq, header):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        confmat.update(labels.flatten(), outputs.argmax(1).flatten())

    confmat.reduce_from_all_processes()
    return confmat



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou