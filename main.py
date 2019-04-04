import argparse
import datetime
import os

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter

from data_loader.data_loader import import_cell_dataset
from model.eval import Eval
from model.loss import calc_loss
from model.train import Trainer
from model.unet import UNet


def main(args: argparse.Namespace):
    cudnn.benchmark = True  # Optimise for hardware
    args.outf = os.path.join(args.outf, datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))

    # Create output directory
    os.makedirs(args.outf)
    if not os.access(os.path.split(args.outf)[0], os.W_OK):  # Check you can write to output path directory
        raise OSError("--model_path is not a writeable path: %s" % args.model_path)

    # Setup logging and tensor board
    writer = SummaryWriter(args.outf)
    writer.add_text("inputs", str(args))
    print(str(args))

    # Define device type
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("device: ", device)

    # Import data loader
    dataset = import_cell_dataset(args.dataroot, crop_size=args.size, edges=args.edges)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True,
                                             num_workers=int(args.nworkers))
    n_channels = dataset.shape[0][0]
    n_classes = 1

    # Import model
    model = UNet(n_channels, n_classes)
    model.to(device)
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location=device))

    # Parameters
    criterion = calc_loss
    # criterion = nn.BCELoss()

    if args.mode == 'train':
        optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        # Train
        trainer = Trainer(model, optimiser, criterion, device, writer, args.outf)
        trainer.train(dataloader, args.niter)

    elif args.mode == 'eval':
        evaluator = Eval(model, criterion, device)
        evaluator.evaluate(dataloader)


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='train | eval')

    parser.add_argument('--dataroot', default='../data/data-science-bowl-2018',
                        help='path to root dataset directory. default: ../data/data-science-bowl-2018')
    parser.add_argument('--batchsize', type=int, default=5, help='number of samples in batch')
    parser.add_argument('--size', type=int, default=256, help='input size of model')
    parser.add_argument('--dataset', default='kaggle', help='mnist or kaggle. default: kaggle')
    parser.add_argument('--nworkers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--disable_cuda', default=False, action='store_true', help='enables cuda')
    parser.add_argument('--model', default=None, type=str, help="path to model (to continue training)")
    parser.add_argument('--outf', default='./saved/runs/', help='folder to output model checkpoints')
    parser.add_argument('--edges', default=False, action='store_true', help='train on edges only')
    args = parser.parse_args()

    main(args)
