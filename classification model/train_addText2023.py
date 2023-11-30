import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random
from read_dataset import classification_dataset, classification_dataset_torchio, classification_dataset_torchio_addText
from utils import read_split_data, create_lr_scheduler, get_params_groups,\
    train_one_epoch, evaluate, plot_data_loader_image, plot_class_preds, test_result, train_one_epoch_addtext, evaluate_addtext, test_result_addtext
from model import convnext_tiny, convnext_tiny_attention, convnext_tiny_weaklySupervised, ResNet3d
from my_dataset import MyDataSet
import torchio as tio
from resnet3D import generate_model
from mobilenet3D import get_model
import sys
from mobilenetv23D import get_model_mobilenetv2, get_model_mobilenetv2_addtext


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'using {device} device')
    output_path = os.path.join(args.data_path, args.strategy)
    if os.path.exists (output_path) is False:
        os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)
    log_dir = os.path.join(output_path, 'runs')
    tb_writer = SummaryWriter(log_dir=log_dir)

    data_transform = {

        "train": tio.Compose([

            tio.Resize((128, 128, 64)),
            tio.RandomFlip(axes=(0, 1, 2), p=0.3),
            tio.RandomAffine(degrees=15, p=0.3),
            tio.RescaleIntensity(out_min_max=(-1, 1), in_min_max=(-100, 200)),])
        ,
        "val": tio.Compose([

            tio.Resize((128, 128, 64)),
            tio.RescaleIntensity(out_min_max=(-1, 1), in_min_max=(-100, 200)),
        ])
    }


    train_dataset = classification_dataset_torchio_addText(args.data_path, classification_json='4_train.json', transform=data_transform['train'])
    val_dataset = classification_dataset_torchio_addText(args.data_path, classification_json='4_test.json', transform=data_transform['val'])
    batch_size =args.batch_size
    nw = 18
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = get_model_mobilenetv2_addtext(num_classes=args.num_classes, sample_size=128, width_mult=1.).to(device)
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch_addtext(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc= evaluate_addtext(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc :
            torch.save(model.state_dict(),
                       f'{output_path}/weights/valacc{val_acc}convnextTwoClassific_bestvalacc.pth')
            best_acc = val_acc

        if epoch == args.epochs - 1:
            end_acc = val_acc
            torch.save(model.state_dict(),
                       f'{output_path}/weights/valacc{end_acc}convnextTwoClassific_end.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--data-path', type=str, default='')
    parser.add_argument('--strategy', type=str, default='AddText_croplabel_mobilnetv2_class2_lr5e-4batch4_resizefirst_fold4')
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch_size', type=int, default= 4)
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--epochs', type=int, default=200)
    opt = parser.parse_args()
    main(opt)

