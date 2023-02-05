import h5py
import math
import pandas as pd
from tensorflow.keras.utils import Sequence
import numpy as np


class ECGSequence(Sequence):
    # https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner
    #
    # @classmethod means: when this method is called, we pass the class as the first argument
    # instead of the instance of that class (as we normally do with methods). This means you can use the class
    # and its properties inside that method rather than a particular instance.
    # The first argument for @classmethod function must always be cls (class).
    #
    # @staticmethod means: when this method is called, we don't pass an instance of the class to it (as we
    # normally do with methods). This means you can put a function inside a class, but you can't access the
    # instance of that class (this is useful when your method does not use the instance).
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        data = pd.read_csv(path_to_csv, usecols=["1dAVb","RBBB","LBBB","SB","ST","AF","trace_file"])
        data.set_index("trace_file", inplace=True)
        data = data.loc['exams_part0.hdf5'].values

        # n_samples = len(pd.read_csv(path_to_csv))
        n_samples = len(data)
        n_train = math.ceil(n_samples * (1 - val_split))  # Round a number upward to its nearest integer
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    # The __init__ function is run once when instantiating the Dataset object.
    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8, start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            # self.y = pd.read_csv(path_to_csv).values    # returning all data of .csv file
            # self.y = pd.read_csv(path_to_csv, usecols=["1dAVb","RBBB","LBBB","SB","ST","AF"]).values
            self.y = pd.read_csv(path_to_csv, usecols=["1dAVb","RBBB","LBBB","SB","ST","AF","trace_file"])
            self.y.set_index("trace_file", inplace=True)
            self.y = self.y.loc['exams_part0.hdf5'].values

        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]  # return tracings (ECG data with shape (N, 4096, 12), N is number of exams_id)
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    # https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python
    @property
    def n_classes(self):
        return self.y.shape[1]  # return number of column in .csv file

    # Method called at the end of every epoch.
    def __getitem__(self, idx):
        """Gets batch at position `idx`.

        Args:
            idx: position of the batch in the Sequence.
        Returns:
            A batch
        """
        # generating index for each batch
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)

        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
