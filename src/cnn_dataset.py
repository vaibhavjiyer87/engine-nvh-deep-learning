import h5py
import numpy as np
import torch

from torch.utils.data import Dataset


class H5LogMelDataset(Dataset):
    """
    PyTorch Dataset for CNN-001 cached log-mel spectrograms.

    Each item returns:
        x_tensor:
            [1, n_mels, time_frames]

        y_tensor:
            standardized RPM target

        cache_index:
            corresponding HDF5 row index
    """

    def __init__(
        self,
        h5_path,
        indices,
        logmel_mean,
        logmel_std,
        rpm_mean,
        rpm_std,
    ):

        self.h5_path = str(
            h5_path
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.logmel_mean = float(
            logmel_mean
        )

        self.logmel_std = float(
            logmel_std
        )

        self.rpm_mean = float(
            rpm_mean
        )

        self.rpm_std = float(
            rpm_std
        )

        self._h5 = None


    def _get_h5(self):

        if self._h5 is None:

            self._h5 = h5py.File(
                self.h5_path,
                "r",
            )

        return self._h5


    def __len__(self):

        return len(
            self.indices
        )


    def __getitem__(
        self,
        dataset_index,
    ):

        h5 = self._get_h5()

        cache_index = int(
            self.indices[
                dataset_index
            ]
        )


        x = (
            h5[
                "log_mel"
            ][
                cache_index
            ]
            .astype(
                np.float32
            )
        )


        rpm = float(
            h5[
                "rpm_mean"
            ][
                cache_index
            ]
        )


        # Input normalization using
        # TRAINING-ONLY statistics.
        x = (
            x
            - self.logmel_mean
        ) / self.logmel_std


        # Target normalization using
        # TRAINING-ONLY statistics.
        y = (
            rpm
            - self.rpm_mean
        ) / self.rpm_std


        # [mel, time] -> [channel, mel, time]
        x_tensor = (
            torch.from_numpy(
                x
            )
            .unsqueeze(0)
        )


        y_tensor = torch.tensor(
            y,
            dtype=torch.float32,
        )


        return (
            x_tensor,
            y_tensor,
            cache_index,
        )


    def close(self):

        if self._h5 is not None:

            self._h5.close()

            self._h5 = None


    def __del__(self):

        self.close()
