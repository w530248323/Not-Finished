import argparse


def parse_opts():
    parser = argparse.ArgumentParser(
        description='Jester Training based on PyTorch ')

    parser.add_argument('--train_data_folder', default='/Users/wangjingyao/Downloads/gesture_dataset/videos',
                        help='train data folder path')
    parser.add_argument('--val_data_folder', default='/Users/wangjingyao/Downloads/gesture_dataset/videos',
                        help='validate data folder path')
    parser.add_argument('--train_labels', default='labels/v1-train.csv',
                        help='train labels file path')
    parser.add_argument('--val_labels', default='labels/v1-validation.csv',
                        help='validate labels file path')
    parser.add_argument('--labels', default='labels/v1-labels.csv',
                        help='labels file path')
    parser.add_argument('--output_dir', default='trainings/',
                        help='output directory')

    parser.add_argument('--num_workers', default=10,
                        help='Number of workers')
    parser.add_argument('--n_classes', default=27,
                        help='Number of classes')
    parser.add_argument('--batch_size', default=64,
                        help='batch size')
    parser.add_argument('--sample_duration', default=18,
                        help='Temporal duration of inputs')
    parser.add_argument('--sample_size', default=84,
                        help='Height and width of inputs')
    parser.add_argument('--nclips', default=1,
                        help='') # ##################
    parser.add_argument('--step_size', default=2,
                        help='') # ###################

    parser.add_argument('--momentum', default=0.9,
                        help='Momentum')
    parser.add_argument('--lr', default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--last_lr', default=0.00001,
                        help='last learning rate')
    parser.add_argument('--weight_decay', default=0.00001,
                        help='Weight Decay')
    parser.add_argument('--n_epochs', default=3000,
                        help='Number of total epochs to run')
    parser.add_argument('--print_freq', default=100,
                        help='print frequency')

    parser.add_argument('--model', '-m', default='resnet',
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=101,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B',
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32,
                        help='ResNeXt cardinality')

    parser.add_argument('--eval_only', '-e',
                        help="evaluate trained model on validation data.")
    parser.add_argument('--resume', '-r',
                        help="resume training from given checkpoint.")
    parser.add_argument('--checkpoint', default=None,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--gpus', '-g', default='0',
                        help="gpu ids for use.")
    parser.add_argument('--pretrain_path', default='',
                        help='Pretrained model (.pth)')

    args = parser.parse_args()

    return args
