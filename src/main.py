# Standard library imports
import random
import time
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# GPU AND CPU KMEANS
from sklearn.cluster import KMeans
from kmeans import kmeans as KMeans_GPU

# Visualization
import matplotlib.pyplot as plt

# Custom imports
from batch_sampler import (
    FewShotBatchSamplerModified,
    FewShotBatchSamplerModifiedV2,
    StratifiedFewShotBatchSamplerBenignCycled
)
from dataset import RoadDataset_v3
from proto_loss import classify_feats, calculate_prototypes, squared_euclidean_distance
from loss.circle_loss import CircleLoss2
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss
from model.model import Conv1dAnomalyTransformer, OnlyAnomalyTransformer
from model.ConvNet import ConvNet
from model.ResNet import ResNet, SupConResNet
from scheduler import CosineWarmupScheduler
import utils as utl

from args import parse_args

# Parse arguments
args = parse_args()

# ============================= SEED AND DEVICE =============================
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)
device = utl.use_cuda(0)
cl_loss = CircleLoss2(margin=0.5).to(device)
# ============================= MODEL HYPERPARAMETERS =============================
# Model architecture
embedding_dims = args.embedding_dims
d_model = args.d_model
layer = args.layers

# Training parameters
batch_size = args.batch_size
epochs = args.epochs
GRADIENT_ACCUMULATION_STEPS = args.grad_accum_steps  # Effective batch size = batch_size * GRADIENT_ACCUMULATION_STEPS

# Dataset configuration
DATASET = args.dataset
ws = args.window_size
s = args.step_size
feat = args.features

# ============================= TRAINING CONFIGURATION =============================
# Few-shot configuration
TRAIN_SHOT = args.train_shot
TEST_SHOT = args.test_shot
EPISODIC_TRAIN = TRAIN_SHOT == 0 and TEST_SHOT == 0

# Model options
NORM = args.norm
OOD_LOSS = args.ood_loss
USE_KMEANS = args.use_kmeans

# Loss configuration
with_cl = args.circle_loss
with_ce = args.center_loss
with_triplet = args.triplet_loss
only_cl = args.only_circle_loss

# Adjust configuration based on dataset
if only_cl:
    with_cl = False
    with_triplet = False

# ============================= PATH CONFIGURATION =============================
# Dataset split configuration
split = args.split
data_split = args.data_split
road_type = f"fab_{split}_split_{data_split}" if DATASET == 1 else 'chd_ss_2_no_split_60test'

# Training configuration string components
optim_type = args.optimizer
scheduler_warmup = args.scheduler_warmup
episodic = "_Episodic" if EPISODIC_TRAIN else ""
norm = "_Norm" if NORM else ""
extra = f"_{TRAIN_SHOT}T{TEST_SHOT}Shot"
additional_loss = ("_withcll2" if with_cl else "_withtriplet" if with_triplet else "")
if OOD_LOSS:
    additional_loss += "_oodloss"

MODEL_NAME = args.model
# Final name construction
NAME = (f'{road_type}_e{embedding_dims}{additional_loss}_'
        f'{MODEL_NAME}_'
        f'{optim_type}_Warmup{scheduler_warmup}'
        f'{f"_{d_model}d{layer}l"}{norm}{episodic}{extra}')

# Debug configuration
DEBUG = args.debug
TQDM = args.tqdm
SAVE_PATH = f'{args.checkpoint_dir}/{NAME}_test_{"debug" if DEBUG else ""}.pth'

print("USE KMEANS", USE_KMEANS)

# ============================= LOAD DATASET =============================

# ============================= DATASET LOADING =============================
N_WAY, K_support, K_query = None, None, None
if EPISODIC_TRAIN:
    N_WAY, K_support, K_query = 5, 5, 15

def create_dataloader(dataset, is_train=False):
    """Create dataloader with appropriate configuration."""
    if is_train and EPISODIC_TRAIN:
        # sampler = StratifiedFewShotBatchSamplerBenignCycled(
        #     dataset_targets=torch.tensor(dataset.y),
        #     N_way=N_WAY,
        #     K_support=K_support,
        #     K_query=K_query
        # )
        sampler = FewShotBatchSamplerModified(
            dataset_targets=torch.tensor(dataset.y),
            N_way=N_WAY,
            K_support=K_support,
            K_query=K_query,
            shuffle=is_train
        )
        return DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)
    else:
        return DataLoader(
            dataset,
            batch_size=256,
            shuffle=is_train,
            num_workers=4,
            pin_memory=True
        )

# Initialize datasets
tr_dataset = RoadDataset_v3(
    dataset=DATASET,
    mode='train',
    road_type=road_type,
    with_time=True,
    embed=d_model,
    new=False,
    seperate_proto=TRAIN_SHOT > 0,
    shot=TRAIN_SHOT,
    ws=ws,
    s=s
)
num_class = len(np.unique(tr_dataset.y))

val_ds = RoadDataset_v3(
    dataset=DATASET,
    mode='val' if args.val_split else 'test',
    road_type=road_type,
    with_time=True,
    embed=d_model,
    seperate_proto=TEST_SHOT > 0,
    shot=TEST_SHOT,
    new=False,
    ws=ws,
    s=s
)

test_ds = RoadDataset_v3(
    dataset=DATASET,
    mode='test',
    road_type=road_type,
    with_time=True,
    embed=d_model,
    seperate_proto=TEST_SHOT > 0,
    shot=TEST_SHOT,
    new=False,
    ws=ws,
    s=s
)

# Create dataloaders
train_loader = create_dataloader(tr_dataset, is_train=True)
val_loader = create_dataloader(val_ds)
test_loader = create_dataloader(test_ds)

USE_OTHER_MODELS = MODEL_NAME == 'ConvNet' or MODEL_NAME == 'ResNet'

# ============================= GET PROTOTYPES =============================
def load_prototypes_from_dataset(dataset):
    """Load prototypes from a dataset and move them to GPU."""
    return (
        dataset.proto_x.to(device, non_blocking=True),
        dataset.proto_y.to(device, non_blocking=True),
        dataset.proto_time.to(device, non_blocking=True)
    )

def initialize_prototypes():
    """Initialize and load prototypes for non-episodic training."""

    # Initialize empty tensors on GPU
    proto_x = torch.tensor([]).to(device, non_blocking=True)
    proto_y = torch.tensor([]).to(device, non_blocking=True)
    proto_t = []

    # Load test prototypes
    if TEST_SHOT > 0:
        proto_x, proto_y, t = load_prototypes_from_dataset(test_ds)
        proto_t = [test_ds.get_proto_time(t[idx].cpu().numpy()) for idx in range(len(t))]

    # Load training prototypes
    if TRAIN_SHOT > 0:
        train_x, train_y, train_t = load_prototypes_from_dataset(tr_dataset)
        proto_x = torch.cat((proto_x, train_x), dim=0)
        proto_y = torch.cat((proto_y, train_y), dim=0)
        proto_t.extend([test_ds.get_proto_time(t.cpu().numpy()) for t in train_t])

    # Stack time prototypes
    proto_t = torch.stack(proto_t).to(device, non_blocking=True)
    return proto_x, proto_y, proto_t

# Initialize prototypes
if (not EPISODIC_TRAIN):
    proto_x, proto_y, proto_t = initialize_prototypes()
else:
    proto_x, proto_y, proto_t = None, None, None

# ============================= MODEL INITIALIZATION =============================
def initialize_model():
    """Initialize model, optimizer, scheduler, and loss functions."""
    # Initialize model
    if (MODEL_NAME == 'Trans'):
        model = OnlyAnomalyTransformer(
            d_model=d_model,
            layer=layer,
            num_class=num_class,
            with_time=True,
            use_emb=True,
            add_norm=NORM,
            in_dim=feat,
            emb_size=embedding_dims,
            win_size=ws
        ).to(device)   
    elif (MODEL_NAME == 'ConvNet'):
        model = ConvNet(x_dim=1, hid_dim=64, z_dim=64).to(device)
    elif (MODEL_NAME == "ResNet"):
        model = SupConResNet().to(device)
    else:
        model = Conv1dAnomalyTransformer(
            d_model=d_model,
            layer=layer,
            num_class=num_class,
            with_time=True,
            use_emb=True,
            add_norm=NORM,
            in_dim=feat,
            emb_size=embedding_dims,
            win_size=ws
        ).to(device)
    print("Init model:", model)
 

    if (args.pretrained_path):
        model.load_state_dict(torch.load(args.pretrained_path))

    # Print model information
    utl.cal_model_size(model)
    utl.print_model_info_pytorch(model)

    # Initialize optimizer and scheduler
    optimizer = utl.init_optim(model, type=optim_type, lr=args.lr)
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=scheduler_warmup,
        max_iters=len(train_loader) * epochs
    )

    # Initialize loss functions
    loss_fns = {
        'cr_loss': nn.CrossEntropyLoss().to(device),
        'cl_loss': CircleLoss2(margin=0.5).to(device),
        'tl_loss': TripletLoss(device).to(device),
        'ce_loss': CenterLoss(num_class, embedding_dims, use_gpu=True).to(device)
    }

    print("Optimizer parameters:", len(optimizer.param_groups[0]['params']))

    return model, optimizer, lr_scheduler, loss_fns

model, optimizer, lr_scheduler, loss_fns = initialize_model()   
# ============================= TRAINING =============================

# ===== Utils Functions =====
def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for sample_batch in dataloader:
            x, y, t = sample_batch
            x, y, t = x.to(device), y.to(device), t.to(device)
            model_output = model(x, t, device) if (not USE_OTHER_MODELS) else model(x)

            embeddings.append(model_output)
            labels.append(y.to(device))
    return torch.cat(embeddings), torch.cat(labels)

def select_support_set_kmeans(embeddings, labels, num_classes, k_shot, n_clusters=5):
    support_set = defaultdict(list)
    unique_labels = torch.unique(labels)
    for label in unique_labels[:num_classes]:
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_indices]

        if (torch.cuda.is_available()):
            kmeans = KMeans_GPU(
                X=class_embeddings, num_clusters=n_clusters, distance='euclidean', device=embeddings.device,
                tqdm_flag=False,
            )
            cluster_centers = kmeans[1]
        else:
            kmeans = KMeans(n_clusters=n_clusters).fit(class_embeddings.cpu().numpy())
            cluster_centers = kmeans.cluster_centers_ 

        selected_indices = []
        for center in cluster_centers:
            if (torch.cuda.is_available()):
                center_idx = ((class_embeddings - center.clone().detach().to(device)).norm(dim=1)).argmin().item()
            else:
                center_idx = ((class_embeddings - torch.tensor(center, device=device)).norm(dim=1)).argmin().item()
            selected_indices.append(class_indices[center_idx])
        selected_indices = random.sample(selected_indices, k_shot)
        for idx in selected_indices:
            support_set[label.item()].append(embeddings[idx])

    
    return support_set

def sample_KMEAN_support(dataset):
    get_proto_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    embs, l = extract_embeddings(model, get_proto_loader)
    support_set = select_support_set_kmeans(embs, l, num_classes=num_class, k_shot=K_support, n_clusters=K_support)
    prototypes = torch.cat([torch.stack(v).mean(dim=0).view(1, -1) for v in support_set.values()], dim=0)

    return prototypes

def sample_global_support(dataset):
    p_x, p_y, t = dataset.proto_x.to(device, non_blocking=True), dataset.proto_y.to(device, non_blocking=True), dataset.proto_time.to(device, non_blocking=True)
    proto_t = []
    for idx, t in enumerate(t):
        proto_t.append(dataset.get_proto_time(t.cpu().numpy()))
    p_t = torch.stack(proto_t).to(device, non_blocking=True)

    return p_x, p_y, p_t

from proto_loss import classify_feats, calculate_prototypes, squared_euclidean_distance

def compute_Lout(D_OOD, centroids, sigma=1.0):
    L_out = 0.0
    for x_s in D_OOD:
        distances = torch.tensor([squared_euclidean_distance(x_s, c) for c in centroids])
        exp_distances = torch.exp(-distances / (2 * sigma ** 2))
        softmax_probs = exp_distances / torch.sum(exp_distances)
        L_out += -torch.log(softmax_probs[0])  # Assuming we're checking against the first class
    
    L_out /= len(D_OOD)  # Average over OOD samples
    return L_out

def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def random_sampling_OOD(labels):
    """Sample OOD data points for training."""
    available_classes = set(range(num_class)) - set(labels.unique().tolist())
    support_indices = []
    
    # Get indices for each available class
    for cls in available_classes:
        cls_indices = torch.where(torch.tensor(tr_dataset.y) == cls)[0]
        support_indices.extend(torch.randperm(len(cls_indices))[:K_support].tolist())
    
    # Collect samples
    ood_samples = [[], [], []]  # x, y, t
    for idx in support_indices:
        x, y, t = tr_dataset.__getitem__(idx)
        for sample_list, item in zip(ood_samples, [x, y, t]):
            sample_list.append(item)
    
    # Move all data to GPU at once
    return [torch.stack(samples).to(device, non_blocking=True) for samples in ood_samples]

class MetricMonitor():
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: 0)
        self.counts = defaultdict(lambda: 0)

    def update(self, metric_name, metric_value):
        self.metrics[metric_name] += metric_value
        self.counts[metric_name] += 1
    
    def get_mean(self, metric_name):
        return self.metrics[metric_name] / self.counts[metric_name]

    def __str__(self):
        return ", ".join([f"{k}: {v / self.counts[k]}" for k, v in self.metrics.items()])
    
# ====== Training ======
def forward_pass(model, data, is_train=True):
    global proto_x, proto_y, proto_t
    
    """Perform forward pass through the model."""
    x, y, t = [item.to(device, non_blocking=True) for item in data]

    # Forward pass
    output = model(x, t, device) if (not USE_OTHER_MODELS) else model(x)
        
    # Get prototypes if needed
    if (EPISODIC_TRAIN and is_train):
        proto_idxs, query_idxs= utl.get_prototypes_indexes(y, N_WAY, K_support)
        # Proto X
        proto_out, proto_y, y = output[proto_idxs], y[proto_idxs], y[query_idxs]
        output = output[query_idxs]
    elif USE_KMEANS:
        proto_out = None
    else: 
        proto_out = model(proto_x, proto_t, device) if (not USE_OTHER_MODELS) else model(proto_x)

    # Get OOD samples if needed
    ood_output = None
    if OOD_LOSS:
        ood_x, _, ood_t = random_sampling_OOD(y)
        if (len(ood_x) != 0):
            ood_output = model(ood_x, ood_t, device) if (not USE_OTHER_MODELS) else model(ood_x)

    return output, proto_out, proto_y, ood_output, y

def evaluate(model, test_loader, current_accuracy=0, proto=None):
    global proto_x, proto_y, proto_t
    
    model.eval()
    metric_monitor = MetricMonitor()
    y_pred = []
    y_true = []

    with torch.no_grad():
        iterator = tqdm(test_loader, desc=f'Validating: ', unit='batch') if TQDM else test_loader
        for i, data in enumerate(iterator):
            x, y, t = data
            x, y, t = x.to(device), y.to(device), t.to(device)

            # Compute predictions
            if (not USE_KMEANS):
                proto_out = model(proto_x, proto_t, device) if (not USE_OTHER_MODELS) else model(proto_x)
                proto_embs, classes = calculate_prototypes(proto_out, proto_y)
            else:
                proto_embs = proto
                classes = torch.tensor(range(num_class)).to(device, non_blocking=True)

            # Forward pass
            output, _, _, _, y = forward_pass(model, data, is_train=False)
            
            preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')

            loss = loss_fns['cr_loss'](preds, labels)
            
            metric_monitor.update('loss', loss.item())
            metric_monitor.update('acc', acc.item())
            metric_monitor.update('f1', f1.item())

            y_true.extend(labels.cpu().numpy())
            preds = preds.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
        
        avg_acc = metric_monitor.get_mean('acc')
        avg_loss = metric_monitor.get_mean('loss')
        if (avg_acc > current_accuracy):
            utl.print_result(y_true, y_pred)
            utl.draw_confusion(y_true, y_pred)
            torch.save(model.state_dict(), SAVE_PATH)
            return avg_acc, avg_loss
        print(f"Validation loss: {metric_monitor}")
    return current_accuracy, avg_loss

def compute_loss(output, y, proto_out=None, proto_y=None, ood_output=None):
    """Compute all relevant losses."""
    # Compute prototypes and predictions
    proto_embs, classes = calculate_prototypes(proto_out, proto_y)
    preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')
    
    # Main classification loss
    cr_loss = loss_fns['cr_loss'](preds, labels)
    loss = cr_loss

    # Additional losses
    if with_cl and proto_out is not None:
        loss += cl_loss(l2_normalize(proto_out), proto_y)
    if with_triplet:
        loss += loss_fns['tl_loss'](l2_normalize(output), y)
    if with_ce and proto_out is not None:
        loss += loss_fns['ce_loss'](l2_normalize(proto_out), proto_y)
    if only_cl:
        loss = loss_fns['cl_loss'](output, y)
    if OOD_LOSS and EPISODIC_TRAIN and ood_output is not None:
        loss += compute_Lout(l2_normalize(ood_output), l2_normalize(proto_out))

    # print("Loss requires_grad:", loss.requires_grad)
    
    return loss, acc, f1, cr_loss

model.train()
train_his = []
best_accuracy = 0
metric = MetricMonitor()

for epoch in range(epochs):
    metric.reset()
    iterator = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch') if TQDM else train_loader
    for i, data in enumerate(iterator):
        x, y, t = data
        x, y, t = x.to(device), y.to(device), t.to(device)
        optimizer.zero_grad()

        # Forward pass
        output, proto_out, proto_y, ood_output, y = forward_pass(model, data, is_train=True)
        
        loss, acc, f1, cr_loss = compute_loss(output, y, proto_out, proto_y, ood_output=ood_output)

        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        metric.update('loss', loss.item())
        metric.update('cr_loss', cr_loss.item())
        metric.update('acc', acc.item())
        metric.update('f1', f1.item())
        
        if i % 200 == 0 and TQDM:
            tqdm.write(f'Batch {i+1}/{len(train_loader)}, Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    train_loss = metric.get_mean('cr_loss')
    print(f"Epoch {epoch}: {metric}")
    
    # Test
    if (EPISODIC_TRAIN):
        get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, new=False, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
        if (USE_KMEANS):
            prototypes = sample_KMEAN_support(get_proto_set)
        else:
            p_x, p_y, p_t = sample_global_support(get_proto_set)
            proto_x, proto_y, proto_t = p_x, p_y, p_t

    temp_best_acc = best_accuracy
    best_accuracy, eval_loss = evaluate(model, val_loader, best_accuracy, prototypes if (USE_KMEANS) else None)
    if (best_accuracy > temp_best_acc and epoch > 100):
        print("====== Evaluate on Test ======")
        if (EPISODIC_TRAIN):
            get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, new=False, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
            if (USE_KMEANS):
                prototypes = sample_KMEAN_support(get_proto_set)
            else:
                p_x, p_y, p_t = sample_global_support(get_proto_set)
                proto_x, proto_y, proto_t = p_x, p_y, p_t

        evaluate(model, test_loader, proto=prototypes if (USE_KMEANS) else None)

    train_his.append((eval_loss, train_loss))

eval_loss, train_loss = zip(*train_his)

utl.plot_his(train_loss, eval_loss, name='Loss', custom_name=NAME)

print("====== Evaluate on Test ======")
if (EPISODIC_TRAIN):
    get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, new=False, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
    if (USE_KMEANS):
        prototypes = sample_KMEAN_support(get_proto_set)
    else:
        p_x, p_y, p_t = sample_global_support(get_proto_set)
        proto_x, proto_y, proto_t = p_x, p_y, p_t

evaluate(model, test_loader, proto=prototypes if (USE_KMEANS) else None)

print("====== Evaluate on Masquerade ======")
if DATASET == 1:
    mar_type = f"mar_{split}_split_{data_split}"
    mar_test = RoadDataset_v3(dataset=1, mode='test', road_type=mar_type, with_time=True, embed=d_model, ws = ws, s = s)
    mar_test_loader = DataLoader(mar_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Test
    if (EPISODIC_TRAIN):
        get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=mar_type, with_time=True, embed=d_model, new=False, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
        if (USE_KMEANS):
            prototypes = sample_KMEAN_support(get_proto_set)
        else:
            p_x, p_y, p_t = sample_global_support(get_proto_set)
            proto_x, proto_y, proto_t = p_x, p_y, p_t

    evaluate(model, mar_test_loader, proto=prototypes if (USE_KMEANS) else None)