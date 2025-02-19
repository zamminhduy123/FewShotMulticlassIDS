# coding=utf-8
import numpy as np
import torch
import random
from collections import defaultdict

class FewShotBatchSampler(object):

    def __init__(self, dataset_targets, N_way, K_support,  K_query=5, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_support = K_support
        self.shuffle = shuffle
        self.K_query = K_query
        self.K_shot = K_support + K_query

        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i+p*self.num_classes for i,
                         c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it*self.N_way:(it+1)*self.N_way]  # Select N classes for the batch

            support_index_batch = []
            query_index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                for _ in range(self.K_support):
                    support_index_batch.append(self.indices_per_class[c][start_index[c]].item())
                    start_index[c] += 1
                    if (start_index[c] == self.indices_per_class[c].shape[0]):
                        start_index[c] = 0
                for _ in range(self.K_query):
                    query_index_batch.append(self.indices_per_class[c][start_index[c]].item())
                    start_index[c] += 1
                    if (start_index[c] == self.indices_per_class[c].shape[0]):
                        start_index[c] = 0
            
            random.shuffle(query_index_batch)
            support_index_batch.extend(query_index_batch)
            yield support_index_batch
            

    def __len__(self):
        return self.iterations
    
class FewShotBatchSamplerModified(object):

    def __init__(self, dataset_targets, N_way, K_support, K_query=5, shuffle=True, shuffle_once=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_support = K_support
        self.shuffle = shuffle
        self.K_query = K_query
        self.K_shot = K_support + K_query

        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        _, counts = torch.unique(self.dataset_targets, return_counts=True)

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide

        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        self.iterations = (int(np.max(list(self.batches_per_class.values()))) // self.N_way)

        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i+p*self.num_classes for i,
                         c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        
        num_class_item = self.batches_per_class.copy()
        self.class_list = []
        for _ in range(0, self.iterations):
            # classes = []
            # while (len(classes) < len(self.classes)):
            #     for c in self.classes:
            #         if num_class_item[c] > 0:
            #             classes.append(c)
            #             num_class_item[c] -= 1
            # self.class_list.extend(classes)
            self.class_list.extend(random.sample(self.classes, self.N_way))

    def __iter__(self):
        if (self.shuffle):
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.iterations):
            support_index_batch = []
            query_index_batch = []
            class_batch = self.class_list[it*self.N_way:(it+1)*self.N_way]  # Select N classes for the batch
            
            for c in class_batch:
                for _ in range(self.K_support):
                    if (start_index[c] >= self.indices_per_class[c].shape[0]):
                        start_index[c] = 0
                    support_index_batch.append(self.indices_per_class[c][start_index[c]].item())
                    start_index[c] += 1
                for _ in range(self.K_query):
                    if (start_index[c] >= self.indices_per_class[c].shape[0]):
                        start_index[c] = 0
                    query_index_batch.append(self.indices_per_class[c][start_index[c]].item())
                    start_index[c] += 1
            random.shuffle(query_index_batch)
            support_index_batch.extend(query_index_batch)
            yield support_index_batch

    def __len__(self):
        return self.iterations
    