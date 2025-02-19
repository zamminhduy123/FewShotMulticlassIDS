import torch
import torch.optim as optim
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix

def create_prototypes_and_queries_with_time(features, targets, time, M, K):
    class_dict = defaultdict(list)
    N = len(targets)
    
    # Group elements by class
    for i in range(N):
        class_dict[targets[i].item()].append((features[i], targets[i], time[i]))
    
    # Ensure we have enough classes
    assert len(class_dict) >= M, "Not enough classes to sample from"

    prototype_features = []
    prototype_targets = []
    prototype_time = []
    query_features = []
    query_targets = []
    query_time = []


    # Sample M unique classes
    sampled_classes = random.sample(list(class_dict.keys()), M)
    
    for cls in sampled_classes:
        # Shuffle and split into support and query sets
        class_samples = class_dict[cls]
        random.shuffle(class_samples)
        
        # Ensure we have enough samples for support set
        assert len(class_samples) >= K, f"Not enough samples in class {cls}"
        
        support_samples = class_samples[:K]
        query_samples = class_samples[K:]

        for sample in support_samples:
            prototype_features.append(sample[0])
            prototype_targets.append(sample[1])
            prototype_time.append(sample[2])
        
        for sample in query_samples:
            query_features.append(sample[0])
            query_targets.append(sample[1])
            query_time.append(sample[2])
    
    prototype_features = torch.stack(prototype_features)
    prototype_targets = torch.stack(prototype_targets)
    query_features = torch.stack(query_features)
    query_targets = torch.stack(query_targets)
    prototype_time = torch.stack(prototype_time)
    query_time = torch.stack(query_time)

    return prototype_features, prototype_targets, prototype_time, query_features, query_targets, query_time


def init_optim(model, type="Adam", lr=0.0001):
    if (type == "Adam"):
        return optim.Adam(model.parameters(), lr=0.0001)
    elif (type == "AdamW"):
        return optim.AdamW(model.parameters(), lr=0.0001)
    else:
        return optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    

def cal_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def use_cuda(gpu):
    # Check if GPU is available
    if torch.cuda.is_available():
        print("Using GPU for training")
    else:
        print("GPU is not available. Using CPU for training")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu)

    print("CUDA", torch.cuda.device_count(), torch.cuda.current_device())
    return device

def print_model_info_pytorch(model):
    """
    Prints all the model parameters, both trainable and non-trainable, and calculates the model size.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        else:
            non_trainable_params += param_size

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    # Calculate model size (assuming 32-bit floats for parameters)
    model_size = total_params * 4 / (1024 ** 2)  # Size in MB (4 bytes per float)
    print(f"Model size: {model_size:.2f} MB")

import matplotlib.pyplot as plt
import datetime
import os
def plot_his(train_his, test_his, name='Loss', custom_name="1"):
    current_time = datetime.datetime.now()
    plt.figure()
    plt.plot(train_his, label=f'Train {name}')
    plt.plot(test_his, label=f'Val {name}')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title(f'{name} History')
    plt.legend()

    # Create directory if it doesn't exist
    os.makedirs('/home/ntmduy/LearningToCompare_FSL/train', exist_ok=True)
    plt.savefig(f'/home/ntmduy/LearningToCompare_FSL/train/{name}_history_{custom_name}.png')

def print_result(label_y, pre_y):
    accuracy = accuracy_score(label_y, pre_y)
    macro_precision = precision_score(label_y, pre_y, average='macro')
    macro_recall = recall_score(label_y, pre_y, average='macro')
    macro_f1 = f1_score(label_y, pre_y, average='macro')
    micro_precision = precision_score(label_y, pre_y, average='micro')
    micro_recall = recall_score(label_y, pre_y, average='micro')
    micro_f1 = f1_score(label_y, pre_y, average='micro')
    print('  -- test result: ')
    print('    -- accuracy: ', accuracy)
    print('    -- macro precision: ', macro_precision)
    print('    -- macro recall: ', macro_recall)
    print('    -- macro f1 score: ', macro_f1)
    print('    -- micro precision: ', micro_precision)
    print('    -- micro recall: ', micro_recall)
    print('    -- micro f1 score: ', micro_f1)
    report = classification_report(label_y, pre_y)
    print(report)

    return accuracy, macro_f1, micro_f1


def draw_confusion(label_y, pre_y):
    cm = confusion_matrix(label_y, pre_y)
    # Calculate the confusion matrix

    # Print False Negatives for each class
    print("False Negatives for each class:")
    for i, label in enumerate(set(label_y)):
        false_negatives = sum(cm[i, :]) - cm[i, i]
        true_positives = cm[i, i]
        if (false_negatives + true_positives) > 0:
            fnr = false_negatives / (false_negatives + true_positives)
        else:
            fnr = 0.0
        print(f"Class {label}: {fnr:.4f}")
    print(cm)