# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from sklearn.metrics import f1_score

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''


    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x, y):
    return (
        F.normalize(x, dim=1)
        @ F.normalize(y, dim=1).T
    )


METRICS = {
    'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
    'euclidean': lambda gallery, query: torch.cdist(query, gallery, p=2),
    'l1': lambda gallery, query: torch.cdist(query, gallery, p=1),
    'l2': lambda gallery, query: torch.cdist(query, gallery, p=2)
}

def calculate_prototypes(features, targets):
    # Given a stack of features vectors and labels, return class prototypes
    # features - shape [N, proto_dim], targets - shape [N]
    classes, _ = torch.unique(targets).sort()  # Determine which classes we have
    prototypes = []
    for c in classes:
        p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
        prototypes.append(p)
    prototypes = torch.stack(prototypes, dim=0)
    # Return the 'classes' tensor to know which prototype belongs to which class
    return prototypes, classes

def squared_euclidean_distance(a, b):
    return torch.sum((a - b) ** 2)

def compute_logits(x, centroids, sigma):
    distances = torch.tensor([squared_euclidean_distance(x, c) for c in centroids])
    logits = -distances / (2 * sigma ** 2)
    return logits

def f1_score_gpu_missing_classes(preds, targets, classes, average='macro', eps=1e-8):
    preds = preds.view(-1)
    targets = targets.view(-1)

    f1_per_class = []
    valid_classes = 0

    for cls in classes:
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        true_positive = torch.sum(pred_cls & target_cls).float()
        false_positive = torch.sum(pred_cls & ~target_cls).float()
        false_negative = torch.sum(~pred_cls & target_cls).float()

        # Skip class clearly if no samples present
        if torch.sum(target_cls) == 0:
            continue

        precision = true_positive / (true_positive + false_positive + eps)
        recall = true_positive / (true_positive + false_negative + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_per_class.append(f1)
        valid_classes += 1

    if valid_classes == 0:
        return torch.tensor(0.0).to(preds.device)

    f1_per_class = torch.stack(f1_per_class)

    if average == 'macro':
        return f1_per_class.mean()
    else:
        raise ValueError("Only 'macro' average supported")
    
def classify_feats(prototypes, classes, feats, targets, metric='euclidean', sigma=1.0):
    # Keep everything on GPU for faster computation
    dist = METRICS[metric](prototypes, feats)
    
    # Apply log softmax to distances (or logits)
    preds = F.log_softmax(-dist, dim=1)
    
    # Since targets are indices within the selected classes, this maps them to the full class set
    labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)

    # Compute metrics on GPU
    with torch.no_grad():
        # Predicted labels are the index of the maximum log probability
        pred_labels = preds.argmax(dim=1)
        
        # Compute accuracy: compare predicted labels with true labels (targets)
        acc = (pred_labels == labels).float().mean()  # Accuracy is the mean of correct predictions

        # Compute F1 score
        f1_score = f1_score_gpu_missing_classes(pred_labels, labels, classes)
    
    return preds, labels, acc, f1_score

def proto_loss_2(query, support, n_classes, n_query):
    query_samples = query.to('cpu')
    prototypes = support.to('cpu')

    dists = METRICS['l1'](prototypes, query_samples)
        # METRICS['l1'](prototypes, query_samples)
    # dists = []
    # dists.append(euclidean_dist(query_samples, prototypes))
    # for metric in METRICS:
    #     dists.append(METRICS[metric](prototypes, query_samples))

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val

def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = METRICS['euclidean'](prototypes, query_samples)
        # METRICS['l1'](prototypes, query_samples)
    # dists = []
    # dists.append(euclidean_dist(query_samples, prototypes))
    # for metric in METRICS:
    #     dists.append(METRICS[metric](prototypes, query_samples))

    log_p_y = F.log_softmax(-dists, dim=1).reshape(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.reshape(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().reshape(-1).mean()
    _, y_hat = log_p_y.max(2)

    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    
    # Calculate F1 score
    y_true = target_inds.squeeze(2).reshape(-1).cpu().numpy()
    y_pred = y_hat.reshape(-1).cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')  # You can choose 'macro', 'micro', or 'weighted' depending on your preference

    return loss_val, acc_val, f1, y_true, y_pred