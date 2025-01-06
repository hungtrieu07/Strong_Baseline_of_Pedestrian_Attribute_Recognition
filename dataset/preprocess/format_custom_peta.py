import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def generate_custom_data_description(save_dir, custom_order):
    """
    Create a dataset description file based on a custom order of attributes.
    """
    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()
    dataset.description = 'custom_peta'
    dataset.reorder = 'custom_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i + 1:05}.png' for i in range(19000)]

    # Extract all attributes and labels
    raw_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]
    raw_label = peta_data['peta'][0][0][0][:, 4:]  # Skip the first 4 columns as before

    # Reorder attributes and labels based on custom_order
    custom_indices = [i - 1 for i in custom_order]  # Convert to zero-based index
    dataset.label = raw_label[:, custom_indices]
    dataset.attr_name = [raw_attr_name[i] for i in custom_indices]

    # Partition information remains the same
    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_train = []
    dataset.weight_trainval = []

    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train.append(weight_train)
        dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, 'custom_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = './data/PETA/'

    # Custom attribute order (indices as provided, adjusted for 1-based indexing)
    custom_order = [
        1, 2, 3, 4, 5, 11, 16, 17, 18, 26, 27, 28, 36, 37, 38, 39, 40, 41, 42, 
        43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 81, 88, 93, 
        94, 97, 99, 102
    ]
    generate_custom_data_description(save_dir, custom_order)
