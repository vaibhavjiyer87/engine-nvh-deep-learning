import h5py
import numpy as np
import torch

from torch.utils.data import Dataset


class HybridLogMelFeatureDataset(Dataset):

    def __init__(
        self,
        h5_path,
        indices,
        engineered_features,
        logmel_mean,
        logmel_std,
        rpm_mean,
        rpm_std,
        torque_mean,
        torque_std,
    ):

        self.h5_path = str(
            h5_path
        )

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.engineered_features = np.asarray(
            engineered_features,
            dtype=np.float32,
        )

        if (
            len(self.engineered_features)
            !=
            len(self.indices)
        ):

            raise ValueError(
                "engineered_features and indices "
                "must contain the same number of rows."
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

        self.torque_mean = float(
            torque_mean
        )

        self.torque_std = float(
            torque_std
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


        logmel = (
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


        torque = float(
            h5[
                "torque_mean_nm"
            ][
                cache_index
            ]
        )


        logmel = (
            logmel
            - self.logmel_mean
        ) / self.logmel_std


        rpm_standardized = (
            rpm
            - self.rpm_mean
        ) / self.rpm_std


        torque_standardized = (
            torque
            - self.torque_mean
        ) / self.torque_std


        logmel_tensor = (
            torch.from_numpy(
                logmel
            )
            .unsqueeze(0)
        )


        feature_tensor = (
            torch.from_numpy(
                self.engineered_features[
                    dataset_index
                ]
            )
        )


        rpm_tensor = torch.tensor(
            rpm_standardized,
            dtype=torch.float32,
        )


        torque_tensor = torch.tensor(
            torque_standardized,
            dtype=torch.float32,
        )


        return (
            logmel_tensor,
            feature_tensor,
            rpm_tensor,
            torque_tensor,
            cache_index,
        )


    def close(self):

        if self._h5 is not None:

            self._h5.close()

            self._h5 = None


    def __del__(self):

        self.close()
