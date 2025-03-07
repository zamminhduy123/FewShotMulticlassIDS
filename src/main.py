# Standard library imports
import random
import time
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
from model.model import Conv1dAnomalyTransformer
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

# ============================= MODEL HYPERPARAMETERS =============================
# Model architecture
embedding_dims = args.embedding_dims
d_model = args.d_model
layer = args.layers
num_class = args.num_class

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
with_ce = args.cross_entropy
with_triplet = args.triplet_loss
only_cl = args.only_circle_loss

# Adjust configuration based on dataset
new = False
if new or DATASET == 0:
    num_class = 3

if only_cl:
    with_cl = False
    with_triplet = False

# ============================= PATH CONFIGURATION =============================
# Dataset split configuration
split = args.split
road_type = f"fab_{split}_split_90test" if DATASET == 1 else 'chd_ss_2_no_split_60test'

# Training configuration string components
optim_type = args.optimizer
scheduler = args.scheduler
episodic = "_Episodic" if EPISODIC_TRAIN else ""
norm = "_Norm" if NORM else ""
extra = f"_{TRAIN_SHOT}T{TEST_SHOT}Shot"
additional_loss = ("_withcll2" if with_cl else "_withtriplet" if with_triplet else "")
if OOD_LOSS:
    additional_loss += "_oodloss"

# Final name construction
NAME = (f'{road_type}_e{embedding_dims}{additional_loss}_'
        f'{"normal" if not new else "2attack"}_{optim_type}_{scheduler}'
        f'{f"_{d_model}d{layer}l"}{norm}{episodic}{extra}')

# Debug configuration
DEBUG = args.debug
TQDM = args.tqdm
SAVE_PATH = f'{args.checkpoint_dir}/{NAME}_Low0_{"debug" if DEBUG else ""}.pth'

# ============================= DATASET LOADING =============================
N_WAY, K_support, K_query = None, None, None
if EPISODIC_TRAIN:
    N_WAY, K_support, K_query = 5, 5, 20

def create_dataloader(dataset, is_train=False):
    """Create dataloader with appropriate configuration."""
    if is_train and EPISODIC_TRAIN:
        sampler = StratifiedFewShotBatchSamplerBenignCycled(
            dataset_targets=torch.tensor(dataset.y),
            N_way=N_WAY,
            K_support=K_support,
            K_query=K_query
        )
        return DataLoader(dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=8,
            pin_memory=True
        )

# Initialize datasets
tr_dataset = RoadDataset_v3(
    dataset=DATASET,
    mode='train',
    road_type=road_type,
    with_time=True,
    embed=d_model,
    new=new,
    seperate_proto=TRAIN_SHOT > 0,
    shot=TRAIN_SHOT,
    ws=ws,
    s=s
)

val_ds = RoadDataset_v3(
    dataset=DATASET,
    mode='val',
    road_type=road_type,
    with_time=True,
    embed=d_model,
    seperate_proto=TEST_SHOT > 0,
    shot=TEST_SHOT,
    new=new,
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
    new=new,
    ws=ws,
    s=s
)

# Create dataloaders
train_loader = create_dataloader(tr_dataset, is_train=True)
val_loader = create_dataloader(val_ds)
test_loader = create_dataloader(test_ds)

# ============================= PROTOTYPE INITIALIZATION =============================
def sample_KMEAN_support(dataset):
    get_proto_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    embs, l = extract_embeddings(model, get_proto_loader)
    support_set = select_support_set_kmeans(embs, l, num_classes=num_class, k_shot=K_support)
    prototypes = torch.cat([torch.stack(v).mean(dim=0).view(1, -1) for v in support_set.values()], dim=0)

    return prototypes, None, None

def sample_global_support(dataset):
    p_x, p_y, t = dataset.proto_x.to(device, non_blocking=True), dataset.proto_y.to(device, non_blocking=True), dataset.proto_time.to(device, non_blocking=True)
    proto_t = []
    for idx, t in enumerate(t):
        proto_t.append(dataset.get_proto_time(t.cpu().numpy()))
    p_t = torch.stack(proto_t).to(device, non_blocking=True)

    return p_x, p_y, p_t
    
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

def load_prototypes_from_dataset(dataset):
    """Load prototypes from a dataset and move them to GPU."""
    return (
        dataset.proto_x.to(device, non_blocking=True),
        dataset.proto_y.to(device, non_blocking=True),
        dataset.proto_time.to(device, non_blocking=True)
    )

# ============================= MODEL INITIALIZATION =============================
def initialize_model():
    """Initialize model, optimizer, scheduler, and loss functions."""
    # Initialize model
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
    ).to(device, non_blocking=True)

    # Print model information
    utl.cal_model_size(model)
    utl.print_model_info_pytorch(model)

    # Initialize optimizer and scheduler
    optimizer = utl.init_optim(model, type=optim_type, lr=args.lr)
    lr_scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=100,
        max_iters=len(train_loader) * epochs
    )

    # Initialize loss functions
    loss_fns = {
        'cr_loss': nn.CrossEntropyLoss().to(device, non_blocking=True),
        'cl_loss': CircleLoss2(margin=0.5).to(device, non_blocking=True),
        'tl_loss': TripletLoss(device).to(device, non_blocking=True),
        'ce_loss': CenterLoss(num_class, embedding_dims, use_gpu=True).to(device, non_blocking=True)
    }

    return model, optimizer, lr_scheduler, loss_fns

# ============================= TRAINING UTILITIES =============================
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

def extract_embeddings(model, dataloader):
    """Extract embeddings from the model for a given dataloader."""
    model.eval()
    embeddings, labels = [], []
    
    with torch.no_grad():
        for x, y, t in dataloader:
            # Move data to GPU
            x, y, t = [item.to(device, non_blocking=True) for item in [x, y, t]]
            
            # Get model output
            model_output = model(x, time=t, device=device)
            embeddings.append(model_output)
            labels.append(y)
    
    return torch.cat(embeddings), torch.cat(labels)

def select_support_set_kmeans(embeddings, labels, num_classes, k_shot, n_clusters=5):
    """Select support set using KMeans clustering."""
    support_set = defaultdict(list)
    unique_labels = torch.unique(labels)
    
    for label in unique_labels[:num_classes]:
        # Get embeddings for current class
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_indices]
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters).fit(class_embeddings.cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        
        # Select samples closest to cluster centers
        selected_indices = []
        for center in cluster_centers:
            center_tensor = torch.tensor(center).to(device, non_blocking=True)
            distances = (class_embeddings - center_tensor).norm(dim=1)
            center_idx = distances.argmin().item()
            selected_indices.append(class_indices[center_idx])
        
        # Adjust number of samples if needed
        if len(selected_indices) < k_shot:
            selected_indices = (selected_indices * (k_shot // len(selected_indices)) + 
                              selected_indices[:k_shot % len(selected_indices)])
        else:
            selected_indices = random.sample(selected_indices, k_shot)
        
        # Add selected samples to support set
        for idx in selected_indices:
            support_set[label.item()].append(embeddings[idx])
    
    return support_set

def compute_Lout(D_OOD, centroids, sigma=1.0):
    """Compute OOD loss."""
    L_out = 0.0
    for x_s in D_OOD:
        distances = torch.tensor([squared_euclidean_distance(x_s, c) for c in centroids])
        exp_distances = torch.exp(-distances / (2 * sigma ** 2))
        softmax_probs = exp_distances / torch.sum(exp_distances)
        L_out += -torch.log(softmax_probs[0])
    return L_out / len(D_OOD)

def l2_normalize(x):
    """Apply L2 normalization to input tensor."""
    return x / x.norm(dim=1, keepdim=True)

# Initialize model and prototypes
model, optimizer, lr_scheduler, loss_fns = initialize_model()

proto_x, proto_y, proto_t = None, None, None
if (not EPISODIC_TRAIN):
    proto_x, proto_y, proto_t = initialize_prototypes()

# ============================= TRAINING AND EVALUATION =============================
def forward_pass(model, data, is_train=True):
    global proto_x, proto_y, proto_t  # Clearly declare globals here

    """Perform forward pass through the model."""
    x, y, t = [item.to(device, non_blocking=True) for item in data]
    
    if is_train and EPISODIC_TRAIN:
        # Sample episode and get prototypes
        p_x, p_y, p_t, x, y, t = utl.create_prototypes_and_queries_with_time(
            x, y, t, M=N_WAY, K=K_support
        )
        proto_x, proto_y, proto_t = [item.to(device, non_blocking=True) for item in [p_x, p_y, p_t]]
        
        # Get OOD samples if needed
        if OOD_LOSS:
            ood_x, ood_y, ood_t = random_sampling_OOD(y)
            combined_x = torch.cat([x, proto_x, ood_x], dim=0)
            combined_t = torch.cat([t, proto_t, ood_t], dim=0)
        else:
            combined_x = torch.cat([x, proto_x], dim=0)
            combined_t = torch.cat([t, proto_t], dim=0)
    else:
        combined_x = torch.cat([x, proto_x], dim=0) if not EPISODIC_TRAIN else x
        combined_t = torch.cat([t, proto_t], dim=0) if not EPISODIC_TRAIN else t
        
    # Forward pass
    combined_output = model(combined_x, combined_t, device)
    
    # Split outputs
    if is_train and EPISODIC_TRAIN:
        output = combined_output[:len(x)]
        proto_out = combined_output[len(x):len(x)+len(proto_x)]
        ood_output = combined_output[len(x)+len(proto_x):] if OOD_LOSS else None
    else:
        output = combined_output[:len(x)]
        proto_out = combined_output[len(x):] if not EPISODIC_TRAIN else None
        ood_output = None
        
    return output, proto_out, ood_output, y, proto_y

def compute_loss(output, y, proto_out=None, proto_y=None, ood_output=None):
    """Compute all relevant losses."""
    # Compute prototypes and predictions
    if proto_out is not None:
        proto_embs, classes = calculate_prototypes(proto_out, proto_y)
        preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')
    else:
        proto_embs, classes = calculate_prototypes(proto_out, proto_y)
        preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')
    
    # Main classification loss
    loss = loss_fns['cr_loss'](preds, labels)
    
    # Additional losses
    if with_cl and proto_out is not None:
        loss += loss_fns['cl_loss'](l2_normalize(proto_out), proto_y)
    if with_triplet:
        loss += loss_fns['tl_loss'](l2_normalize(output), y)
    if with_ce and proto_out is not None:
        loss += loss_fns['ce_loss'](l2_normalize(proto_out), proto_y)
    if only_cl:
        loss = loss_fns['cl_loss'](output, y)
    if OOD_LOSS and EPISODIC_TRAIN and ood_output is not None:
        loss += compute_Lout(l2_normalize(ood_output), l2_normalize(proto_out))
    
    return loss, acc, f1

def evaluate(model, data_loader, current_accuracy=0, proto=None):
    global proto_x, proto_y, proto_t  # Declare global variables
    
    """Evaluate model on given data loader."""
    model.eval()
    metrics = {'loss': [], 'acc': [], 'f1': [], 'y_pred': [], 'y_true': []}
    
    with torch.no_grad():
        for data in data_loader:
            # Forward pass
            output, proto_out, _, y, _ = forward_pass(model, data, is_train=False)
            
            # Compute predictions
            if proto is None:
                proto_out = model(proto_x, proto_t, device)
                proto_embs, classes = calculate_prototypes(proto_out, proto_y)
            else:
                proto_embs = proto
                classes = torch.tensor(range(num_class)).to(device, non_blocking=True)
            
            preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')
            loss = loss_fns['cr_loss'](output, y)
            
            # Record metrics
            metrics['loss'].append(loss.item())
            metrics['acc'].append(acc.item())
            metrics['f1'].append(f1.item())
            metrics['y_true'].extend(labels.cpu().numpy())
            metrics['y_pred'].extend(preds.argmax(dim=1).cpu().numpy())
    
    # Calculate averages
    avg_metrics = {k: np.mean(v) if k != 'y_true' and k != 'y_pred' else v 
                  for k, v in metrics.items()}
    
    # Save model if accuracy improved
    if avg_metrics['acc'] > current_accuracy:
        utl.print_result(avg_metrics['y_true'], avg_metrics['y_pred'])
        utl.draw_confusion(avg_metrics['y_true'], avg_metrics['y_pred'])
        torch.save(model.state_dict(), SAVE_PATH)
        return avg_metrics['acc'], avg_metrics['loss']
    
    print(f"Validation loss: {avg_metrics['loss']}, acc: {avg_metrics['acc']}, f1: {avg_metrics['f1']}")
    return current_accuracy, avg_metrics['loss']

def train_epoch(model, train_loader, optimizer, epoch):
    
    
    """Train model for one epoch."""
    model.train()
    metrics = {'loss': [], 'acc': [], 'f1': []}
    
    iterator = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch') if TQDM else train_loader
    for batch_idx, data in enumerate(iterator):
        optimizer.zero_grad()
        
        # Forward pass
        output, proto_out, ood_output, y, proto_y = forward_pass(model, data)
        
        # Compute loss
        loss, acc, f1 = compute_loss(output, y, proto_out, proto_y, ood_output)
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Record metrics
        metrics['loss'].append(loss.item())
        metrics['acc'].append(acc.item())
        metrics['f1'].append(f1.item())
        
        if batch_idx % 200 == 0 and TQDM:
            tqdm.write(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    # Calculate averages
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"Epoch {epoch} loss: {avg_metrics['loss']}, acc: {avg_metrics['acc']}, f1: {avg_metrics['f1']}")
    
    return avg_metrics

# ============================= MAIN TRAINING LOOP =============================
def train():
    """Main training loop."""
    train_his = []
    best_accuracy = 0

    global proto_x, proto_y, proto_t  # Declare global variables
    
    for epoch in range(epochs):
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, epoch)
        
        # Evaluate
        prototypes = None
        if (EPISODIC_TRAIN):
            get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, 
            new=new, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
            if (USE_KMEANS):
                prototypes, _, _ = sample_KMEAN_support(get_proto_set)
            else:
                proto_x, proto_y, proto_t = sample_global_support(get_proto_set)
        
        best_accuracy, eval_loss = evaluate(model, val_loader, best_accuracy, proto=prototypes)
        
        # Test periodically
        if epoch % 20 == 0:
            evaluate(model, test_loader, best_accuracy, proto=prototypes)
        
        train_his.append((eval_loss, train_metrics['loss']))
    
    return train_his

# Start training
train_history = train()
eval_loss, train_loss = zip(*train_history)
utl.plot_his(train_loss, eval_loss, name='Loss', custom_name=NAME)

# Final evaluation
print("====== Evaluate on Test ======")
evaluate(model, test_loader)

if not new and DATASET == 1:
    print("====== Evaluate on Different Dataset ======")
    mar_type = f"mar_{split}_split_90test"
    mar_test = RoadDataset_v3(
        dataset=1,
        mode='test',
        road_type=mar_type,
        with_time=True,
        embed=d_model,
        ws=ws,
        s=s
    )
    mar_test_loader = DataLoader(mar_test, batch_size=batch_size, shuffle=False, num_workers=8)
    evaluate(model, mar_test_loader)