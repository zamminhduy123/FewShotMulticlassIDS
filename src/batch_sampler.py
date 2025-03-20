# coding=utf-8
import numpy as np
import torch
import random
from collections import defaultdict
from itertools import cycle, islice

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

        # self.batches_per_class[0] = 0
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
    
class FewShotBatchSamplerModifiedV2:
    def __init__(self, dataset_targets, N_way, K_support, K_query=5, shuffle=True, shuffle_once=False):
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_support = K_support
        self.K_query = K_query
        self.K_shot = K_support + K_query
        self.shuffle = shuffle
        self.shuffle_once = shuffle_once

        self.classes = torch.unique(self.dataset_targets).tolist()
        self.indices_per_class = {
            c: torch.where(self.dataset_targets == c)[0].tolist()
            for c in self.classes
        }

        # min_samples = min(len(indices) for indices in self.indices_per_class.values())
        self.iterations = max(len(indices) for indices in self.indices_per_class.values()) // self.K_shot

        # Shuffle ONLY ONCE if specified
        if self.shuffle_once:
            for c in self.classes:
                random.shuffle(self.indices_per_class[c])

    def __iter__(self):
        if self.shuffle:
            for c in self.classes:
                random.shuffle(self.indices_per_class[c])

        # Cycle through classes with fewer samples repeatedly
        class_iterators = {
            c: cycle(self.indices_per_class[c])
            for c in self.classes
        }

        for _ in range(self.iterations):
            selected_classes = random.sample(self.classes, self.N_way)
            batch = []

            for c in selected_classes:
                indices_iter = class_iterators[c]

                support = list(islice(indices_iter, self.K_support))
                query = list(islice(indices_iter, self.K_query))

                batch.extend(support + query)

            yield batch

    def __len__(self):
            return self.iterations

from torch.utils.data import Sampler
class StratifiedFewShotBatchSamplerBenignCycled(Sampler):
    def __init__(self, dataset_targets, N_way, K_support, K_query):
        super().__init__(dataset_targets)
        self.targets = dataset_targets
        self.N_way = N_way
        self.K_support = K_support
        self.K_query = K_query
        self.K_shot = K_support + K_query
        
        self.classes = torch.unique(self.targets).tolist()
        
        # Separate benign class explicitly
        self.benign_class = 0
        self.other_classes = [cls for cls in self.classes if cls != self.benign_class]
        
        # Indices per class
        self.indices_benign = (self.targets == self.benign_class).nonzero(as_tuple=True)[0].tolist()
        self.indices_other = {
            cls: (self.targets == cls).nonzero(as_tuple=True)[0].tolist()
            for cls in self.other_classes
        }
        
        # Determine episodes per epoch from LARGEST non-benign class
        self.max_samples_non_benign = max(len(indices) for indices in self.indices_other.values())
        self.episodes_per_epoch = self.max_samples_non_benign // self.K_shot

    def __iter__(self):
        # Shuffle once per epoch
        shuffled_indices = {
            cls: random.sample(indices, len(indices))
            for cls, indices in self.indices_other.items()
        }

        # Cycle smaller classes explicitly
        iterators = {
            cls: cycle(shuffled_indices[cls])
            for cls in self.other_classes
        }

        for episode in range(self.episodes_per_epoch):
            batch_indices = []

            # Randomly sample benign class
            benign_samples = random.sample(self.indices_benign, self.K_shot)
            batch_indices.extend(benign_samples)

            # Select N_way - 1 other classes randomly
            selected_classes = random.sample(self.other_classes, self.N_way - 1)

            for cls in selected_classes:
                cls_iterator = iterators[cls]
                cls_samples = list(islice(cls_iterator, self.K_shot))
                batch_indices.extend(cls_samples)

            yield batch_indices

    def __len__(self):
        return self.episodes_per_epoch