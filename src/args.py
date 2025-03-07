import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Few-Shot CAN Bus Anomaly Detection')
    
    # Model architecture
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embedding-dims', type=int, default=64, help='Embedding dimensions')
    model_group.add_argument('--d-model', type=int, default=10, help='Model dimension')
    model_group.add_argument('--layers', type=int, default=10, help='Number of layers')
    model_group.add_argument('--num-class', type=int, default=12, help='Number of classes')
    model_group.add_argument('--norm', action='store_true', default=True, help='Use normalization')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch-size', type=int, default=128, help='Batch size')
    train_group.add_argument('--epochs', type=int, default=600, help='Number of epochs')
    train_group.add_argument('--grad-accum-steps', type=int, default=1, help='Gradient accumulation steps')
    train_group.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    train_group.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'], help='Optimizer type')
    train_group.add_argument('--scheduler', type=str, default='CosineWarmup100', help='Learning rate scheduler')
    train_group.add_argument('--debug', action='store_true' , default=True, help='Enable debug mode')
    train_group.add_argument('--tqdm', action='store_true', default=True,  help='Show progress bar')
    
    # Dataset configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=int, default=1, choices=[0, 1], help='Dataset type (0: CarHacking, 1: Road)')
    data_group.add_argument('--window-size', type=int, default=16, help='Window size')
    data_group.add_argument('--step-size', type=int, default=1, help='Step size')
    data_group.add_argument('--features', type=int, default=11, help='Number of features')
    data_group.add_argument('--split', type=str, default='ss_11_2d_no', help='Data split version')
    
    # Few-shot configuration
    fewshot_group = parser.add_argument_group('Few-shot Configuration')
    fewshot_group.add_argument('--train-shot', type=int, default=0, help='Number of training shots')
    fewshot_group.add_argument('--test-shot', type=int, default=0, help='Number of testing shots')
    fewshot_group.add_argument('--use-kmeans', action='store_true', help='Use KMeans for prototype selection')
    
    # Loss configuration
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument('--circle-loss', action='store_true', default=True, help='Use circle loss')
    loss_group.add_argument('--cross-entropy', action='store_true', help='Use cross entropy loss')
    loss_group.add_argument('--triplet-loss', action='store_true', help='Use triplet loss')
    loss_group.add_argument('--only-circle-loss', action='store_true', help='Use only circle loss')
    loss_group.add_argument('--ood-loss', action='store_true', default=True, help='Use OOD loss')
    
    # Paths
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument('--checkpoint-dir', type=str, default='/home/ntmduy/FewShotCanBus/checkpoint',
                           help='Directory to save checkpoints')
    path_group.add_argument('--pretrained-path', type=str, 
                           default='/home/ntmduy/LearningToCompare_FSL/road/trained_model/fab_ss_2_no_split_e64_withcll2_oodloss_normal_AdamW_CosineWarmup100_10d10l_Norm_5T0Shot.pth',
                           help='Path to pretrained model')
    
    args = parser.parse_args()
    
    # Derived configurations
    args.episodic_train = args.train_shot == 0 and args.test_shot == 0
    if args.only_circle_loss:
        args.circle_loss = False
        args.triplet_loss = False
    
    return args