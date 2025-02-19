import time
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd


from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from batch_sampler import FewShotBatchSamplerModified

from dataset import RoadDataset_v3
from model.model import Conv1dAnomalyTransformer

import utils as utl

from loss.circle_loss import CircleLoss2
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss

from collections import defaultdict
from sklearn.cluster import KMeans
from scheduler import CosineWarmupScheduler

# ============================= INITIALIZATION =============================

torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)
device = utl.use_cuda(0)

# ============================= HYPERPARAMETERS =============================

# 1 for Road, 0 for CarHacking
DATASET = 1                                                                    # Adjustable dataset

embedding_dims = 64
batch_size = 64                                                                 # Adjustable bs
epochs = 300                                                          # Adjustable epochs
d_model = 10
layer = 10

# EPISODIC TRAINING is training with different prototypes every epochs for generalization
TRAIN_SHOT = 0
TEST_SHOT = 0
EPISODIC_TRAIN = TRAIN_SHOT == 0 and TEST_SHOT == 0

# window size and steps
ws = 16
s = 16
feat = 9

# Num class and adjustable configurations
num_class = 7
NORM = True
OOD_LOSS = True
USE_KMEANS = True

# Ablation test configurations
# For the final model, we don't use any of  these configurations
new = False                                                                     # Ablation
if (new or DATASET == 0): numclass = 3

with_cl = True                                                                   # Add/Remove circle loss
with_ce = False                                                                   # Add/Remove cross_entropy loss              
only_cl = False                                                                  # Add/Remove only using circle loss
with_triplet = False                                                             # Add/Remove triplet loss
if (only_cl):
    with_cl = False
    with_triplet = False

# ============================= SAVE PATH CONFIGURATION =============================

split = "ss_2_no"                                                                           # version of data split
optim_type = "AdamW"                                                                        # Type of optimizer
scheduler = "CosineWarmup100"                                                               # Type of scheduler
episodic = "_Episodic" if (EPISODIC_TRAIN) else ""                                          # Type of training
norm = "_Norm" if (NORM) else ""                                                            # Using normalization or not
extra = f"_{TRAIN_SHOT}T{TEST_SHOT}Shot"                                                    # Number of shots
additional_loss = "_withcll2" if (with_cl) else "_withtriplet" if (with_triplet) else ""    # Additional loss
if (OOD_LOSS): additional_loss += "_oodloss"

road_type = f"fab_{split}_split_40test"                                                     # The pre-splitted dataset name
if (DATASET == 0):
    road_type = 'chd_ss_2_no_split_95test'
NAME = f'{road_type}_e{embedding_dims}{additional_loss}_{"normal" if (not new) else "2attack"}_{optim_type}_{scheduler}{f"_{d_model}d{layer}l"}{norm}{episodic}{extra}'
SAVE_PATH = f'/home/ntmduy/LearningToCompare_FSL/road/trained_model/{NAME}.pth'

# ============================= LOAD DATASET =============================

tr_dataset = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, new=new, seperate_proto=TRAIN_SHOT > 0, shot = TRAIN_SHOT, ws = ws, s = s)
if (EPISODIC_TRAIN):
    N_WAY, K_support, K_query = 3, 5, 10
    train_fs_sampler = FewShotBatchSamplerModified(torch.tensor(tr_dataset.y), N_way=N_WAY, K_support=K_support, K_query=K_query)
    train_loader = DataLoader(tr_dataset, batch_sampler=train_fs_sampler)
else:
    train_loader = DataLoader(tr_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

test_ds = RoadDataset_v3(dataset=DATASET, mode='test', road_type=road_type, with_time=True, embed=d_model, seperate_proto=TEST_SHOT > 0, shot = TEST_SHOT, new=new, ws = ws, s = s)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# ============================= GET PROTOTYPES =============================
if (not EPISODIC_TRAIN):
    proto_x = torch.tensor([]).to(device)
    proto_y = torch.tensor([]).to(device)
    proto_t = []
    if (TEST_SHOT > 0):
        proto_x, proto_y, t = test_ds.proto_x.to(device), test_ds.proto_y.to(device), test_ds.proto_time.to(device)
        proto_t = []
        for idx, t in enumerate(t):
            proto_t.append(test_ds.get_proto_time(t.cpu().numpy()))

    if (TRAIN_SHOT > 0):
        proto_x, proto_y, t = torch.cat((proto_x, tr_dataset.proto_x.to(device)), dim = 0), torch.cat((proto_y, tr_dataset.proto_y.to(device)), dim = 0), tr_dataset.proto_time.to(device)
        for idx, t in enumerate(t):
            proto_t.append(test_ds.get_proto_time(t.cpu().numpy()))
    proto_t = torch.stack(proto_t).to(device)

# ============================= MODEL INITIALIZATION =============================

model = Conv1dAnomalyTransformer(d_model = d_model, layer = layer, num_class = num_class, with_time = True, use_emb=True, add_norm=NORM, in_dim=feat, emb_size=embedding_dims, win_size=ws).to(device)
utl.cal_model_size(model)
utl.print_model_info_pytorch(model)

utl.cal_model_size(model)
optimizer = utl.init_optim(model, type=optim_type, lr=0.0001)
lr_scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=len(train_loader) * epochs)

cr_loss = torch.nn.CrossEntropyLoss().to(device)
cl_loss = CircleLoss2(margin=0.5).to(device)
tl_loss = TripletLoss(device).to(device)
ce_loss = CenterLoss(num_class, embedding_dims, use_gpu=True).to(device)


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
            model_output = model(x, time=t, device=device)

            embeddings.append(model_output)
            labels.append(y.to(device))
    return torch.cat(embeddings), torch.cat(labels)

def select_support_set_kmeans(embeddings, labels, num_classes, k_shot, n_clusters=5):
    support_set = defaultdict(list)
    unique_labels = torch.unique(labels)
    for label in unique_labels[:num_classes]:
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_embeddings = embeddings[class_indices]
        kmeans = KMeans(n_clusters=n_clusters).fit(class_embeddings.cpu().numpy())
        cluster_centers = kmeans.cluster_centers_
        selected_indices = []
        for center in cluster_centers:
            center_idx = ((class_embeddings - torch.tensor(center).to(device)).norm(dim=1)).argmin().item()
            selected_indices.append(class_indices[center_idx])
        selected_indices = random.sample(selected_indices, k_shot)
        for idx in selected_indices:
            support_set[label.item()].append(embeddings[idx])

    
    return support_set

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

# ====== Training ======

def evaluate(model, test_loader, current_accuracy=0, proto=None):
    model.eval()
    running_loss = []
    running_acc = []
    running_f1 = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x, y, t = data
            x, y, t = x.to(device), y.to(device), t.to(device)
            
            output = model(x, 
                           t, device, classifier=False
                           )

            if (proto is None):
                proto_out = model(proto_x, 
                                  proto_t, device, classifier=False
                                  )
                proto_embs, classes = calculate_prototypes(proto_out, proto_y)
            else:
                proto_embs = proto
                classes = torch.tensor(range(num_class)).to(device)
            preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')

            loss = cr_loss(output, y)
            
            running_loss.append(loss.item())
            running_acc.append(acc.item())
            running_f1.append(f1.item())
            y_true.extend(labels.cpu().numpy())
            preds = preds.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
        
        
        avg_acc = np.mean(running_acc)
        avg_loss = np.mean(running_loss)
        if (avg_acc > current_accuracy):
            utl.print_result(y_true, y_pred)
            utl.draw_confusion(y_true, y_pred)
            # torch.save(model.state_dict(), SAVE_PATH)
            return avg_acc, avg_loss
        print(f"Validation loss: {np.mean(running_loss)}, acc: {avg_acc}, f1: {np.mean(running_f1)}")
    return current_accuracy, avg_loss

model.train()
train_his = []
best_accuracy = 0

for epoch in tqdm(range(epochs), desc="Epochs"):
    running_loss = []
    running_acc = []
    running_f1 = []
    for i, data in enumerate(train_loader, 0):
        x, y, t = data
        x, y, t = x.to(device), y.to(device), t.to(device)
        optimizer.zero_grad()

        if (EPISODIC_TRAIN):
            proto_x, proto_y, proto_t, x, y, t = utl.create_prototypes_and_queries_with_time(x, y, t,M=N_WAY, K=K_support)
            if (OOD_LOSS):
                ood_x, ood_y, ood_t = [], [], []
                # Randomly sample K_support data points from the dataset that have a different class label than y
                available_classes = set(range(num_class)) - set(y.unique().tolist())
                support_indices = []
                for cls in available_classes:
                    cls_indices = torch.where(torch.tensor(tr_dataset.y) == cls)[0]
                    support_indices.extend(torch.randperm(len(cls_indices))[:K_support].tolist())
                for idx in support_indices:
                    ox, oy, ot = tr_dataset.__getitem__(idx)
                    ood_x.append(ox)
                    ood_y.append(oy)
                    ood_t.append(ot)
                ood_x = torch.stack(ood_x)
                ood_y = torch.stack(ood_y)
                ood_t = torch.stack(ood_t)
        
        output = model(x
                       , t, device
                       )
        
        if (OOD_LOSS):
            ood_output = model(ood_x.to(device), 
                               ood_t.to(device), device
                               )

        proto_out = model(proto_x
                          , proto_t, device
                          )

        if (with_cl): ci_loss = cl_loss(l2_normalize(proto_out), proto_y)
        if (with_triplet): tr_loss = tl_loss(l2_normalize(output), y)
        if (with_ce): ct_loss = ce_loss(l2_normalize(proto_out), proto_y)

        proto_embs, classes = calculate_prototypes(proto_out, proto_y)
        preds, labels, acc, f1 = classify_feats(proto_embs, classes, output, y, metric='euclidean')

        # CrossEntropyLoss
        loss = cr_loss(preds, labels)
        # loss = -torch.log(preds).mean()
        if (with_cl): loss += ci_loss
        if (with_triplet): loss += tr_loss
        if (with_ce): loss += ct_loss
        if (only_cl):
            ci_loss = cl_loss(output, y)
            loss = ci_loss

        if (OOD_LOSS):
            loss += compute_Lout(l2_normalize(ood_output), l2_normalize(proto_out))

        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        running_loss.append(loss.item())
        running_acc.append(acc.item()) 
        running_f1.append(f1.item())

        if (i%200 == 0):
            print("\nCurrent loss: ", loss.item())

    train_loss = np.mean(running_loss)
    print(f"Epoch {epoch} loss: {train_loss}, acc: {np.mean(running_acc)}, f1: {np.mean(running_f1)}")
    
    # Test
    if (EPISODIC_TRAIN):
        get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=road_type, with_time=True, embed=d_model, new=new, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
        if (USE_KMEANS):
            get_proto_loader = DataLoader(get_proto_set, batch_size=batch_size, shuffle=False, num_workers=4)
            embs, l = extract_embeddings(model, get_proto_loader)
            support_set = select_support_set_kmeans(embs, l, num_classes=num_class, k_shot=K_support)
            prototypes = torch.cat([torch.stack(v).mean(dim=0).view(1, -1) for v in support_set.values()], dim=0)
        else:
            proto_x, proto_y, t = get_proto_set.proto_x.to(device), get_proto_set.proto_y.to(device), get_proto_set.proto_time.to(device)
            proto_t = []
            for idx, t in enumerate(t):
                proto_t.append(get_proto_set.get_proto_time(t.cpu().numpy()))
            proto_t = torch.stack(proto_t).to(device)

    best_accuracy, eval_loss = evaluate(model, test_loader, best_accuracy, prototypes if (USE_KMEANS) else None)
    train_his.append((eval_loss, train_loss))

eval_loss, train_loss = zip(*train_his)

utl.plot_his(train_loss, eval_loss, name='Loss', custom_name=NAME)

print("====== Evaluate on Mar ======")
if not new and DATASET == 1:
    mar_type = f"mar_{split}_split_40test"
    mar_test = RoadDataset_v3(dataset=1, mode='test', road_type=mar_type, with_time=True, embed=d_model, ws = ws, s = s)
    mar_test_loader = DataLoader(mar_test, batch_size=batch_size, shuffle=False, num_workers=4)

    # Test
    if (EPISODIC_TRAIN):
        get_proto_set = RoadDataset_v3(dataset=DATASET, mode='train', road_type=mar_type, with_time=True, embed=d_model, new=new, seperate_proto=False if (USE_KMEANS) else True, shot = 5 if (USE_KMEANS) else K_support, ws = ws, s = s)
        if (USE_KMEANS):
            get_proto_loader = DataLoader(get_proto_set, batch_size=batch_size, shuffle=False, num_workers=4)
            embs, l = extract_embeddings(model, get_proto_loader)
            support_set = select_support_set_kmeans(embs, l, num_classes=num_class, k_shot=K_support)
            prototypes = torch.cat([torch.stack(v).mean(dim=0).view(1, -1) for v in support_set.values()], dim=0)
        else:
            proto_x, proto_y, t = get_proto_set.proto_x.to(device), get_proto_set.proto_y.to(device), get_proto_set.proto_time.to(device)
            proto_t = []
            for idx, t in enumerate(t):
                proto_t.append(get_proto_set.get_proto_time(t.cpu().numpy()))
            proto_t = torch.stack(proto_t).to(device)

    evaluate(model, mar_test_loader, proto=prototypes if (USE_KMEANS) else None)