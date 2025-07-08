import os
from PIL import Image
import random
from torch.utils.data import Subset

# Split function to perform controlled random split from each protocol file
def split_protocol_dataset(dataset, protocol_entries, train_ratio=0.7):
    train_indices = []
    val_indices = []

    offset = 0
    for proto_entries in protocol_entries:
        num_samples = len(proto_entries)
        indices = list(range(num_samples))
        random.shuffle(indices)

        train_count = int(train_ratio * num_samples)
        proto_train = indices[:train_count]
        proto_val = indices[train_count:]

        train_indices += [offset + idx for idx in proto_train]
        val_indices += [offset + idx for idx in proto_val]

        offset += num_samples

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    return train_set, val_set