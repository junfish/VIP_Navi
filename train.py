import os
import torch
import argparse
from data_loader import get_loader
from solver import Solver
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1996)
    # numpy.random.seed(opt.seed)
    # random.seed(124)
    # torch.backends.cudnn.enabled = False
    data_loader = get_loader(config.model, config.proj_path, config.metadata_path, config.mode, config.geometric, config.batch_size,
                             config.shuffle)
    solver = Solver(data_loader, config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

# python train.py --model Resnet34 --bayesian True --dropout_rate 0.5
# python train.py --model MobilenetV3 --bayesian True --dropout_rate 0.5

# python train.py --model Resnet34 --sx 0 --sq -5 --bayesian False --dropout_rate 0.0 --learn_beta True
# python train.py --model MobilenetV3 --sx 0 --sq -5 --bayesian False --dropout_rate 0.0 --learn_beta True

# python train.py --model Resnet34 --geometric True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_euler6', type=bool, default=False)
    parser.add_argument('--bayesian', type=bool, default=False, help='Bayesian Posenet, True or False')
    parser.add_argument('--geometric', type=bool, default=False, help='Geometric Posenet, True or False')
    parser.add_argument('--sequential_mode', type=str, default=None,
                        choices=[None, 'model', 'fixed_weight', 'batch_size', 'learning_rate', 'beta'])

    parser.add_argument('--lr', type=float, default=0.0001) # 0.0001
    parser.add_argument('--sx', type=float, default=0.0)
    parser.add_argument('--sq', type=float, default=-5.0)
    parser.add_argument('--learn_beta', type=bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='range 0.0 to 1.0')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--fixed_weight', type=bool, default=False)
    parser.add_argument('--model', type=str, default='Resnet50', choices=['Googlenet', 'Resnet', 'Resnet34', 'Resnet50', 'Resnet101', 'Renet34Simple', 'MobilenetV3',
                                                                          'Resnet34lstm', 'MobilenetV3lstm',
                                                                          'Resnet34hourglass', "MobilenetV3hourglass",
                                                                          "Branchresnet34", "BranchmobilenetV3"])
    parser.add_argument('--pretrained_model', type=str, default=None)

    # parser.add_argument('--proj_path', type=str, default='/mnt/data2/image_based_localization/posenet/Street')
    # parser.add_argument('--proj_path', type = str, default='/mnt/data2/NCLT')
    parser.add_argument('--proj_path', type=str, default='/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement')
    # Basement
    # Lower_Floor
    # Floor_3


    # parser.add_argument('--metadata_path', type=str, default='/mnt/data2/image_based_localization/posenet/Street/dataset_train.txt')
    # parser.add_argument('--metadata_path', type=str, default='/mnt/data2/NCLT/train_1m.txt')
    # parser.add_argument('--metadata_path', type=str, default='/mnt/data2/complex_urban/urban08/image_convert/train.txt')
    parser.add_argument('--metadata_path', type=str, default='/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Basement/image_train_all.txt')
    # geometric_data.pkl
    # image_train_all.txt
    # Lower_Floor/image_train_80.txt

    # Training settings
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 1 2 3') # selection of gpu id (single gpu)
    # parser.add_argument('--dataset', type=str, default='Oxford', choices=['NCLT', 'VKITTI', 'Oxford', 'QUT'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16) # 16
    parser.add_argument('--num_workers', type=int, default=1)

    # Test settings
    parser.add_argument('--test_model', type=str, default='29_3000')

    # Misc

    parser.add_argument('--use_tensorboard', type=bool, default=False) # False is set to use wandb

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=50)

    config = parser.parse_args()
    main(config)
