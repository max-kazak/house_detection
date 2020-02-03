import os

import numpy as np
from skimage.io import imread
from sklearn.utils import shuffle
import tensorflow as tf
from keras.utils import Sequence

from src.models.augmentations import resize_only


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir=r'./data/processed/train/', image_folder='images/', mask_folder='masks/',
                 batch_size=1, image_size=1472, nb_y_features=1,
                 augmentation=None,
                 shuffle=True):

        self.image_filenames = [os.path.join(root_dir, image_folder, filename)
                                for filename in os.listdir(os.path.join(root_dir, image_folder))]
        self.mask_names = [os.path.join(root_dir, mask_folder, filename)
                           for filename in os.listdir(os.path.join(root_dir, mask_folder))]
        self.batch_size = batch_size
        self.currentIndex = 0
        self.image_size = image_size
        self.augmentation = augmentation(image_size) if augmentation is not None else resize_only(image_size)
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.shuffle = shuffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name):
        return (imread(image_name) / 255).astype(np.float32), (imread(mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        """
        Generate one batch of data

        """
        # Generate indexes of the batch
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        # Defining dataset
        X = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):
            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])
            # Augmentation code
            augmented = self.augmentation(image=X_sample, mask=y_sample)
            image_augm = augmented['image']
            mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
            X[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
            y[i, ...] = mask_augm

        return X, y