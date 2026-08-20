import torch
import torch.nn as nn


class FrequencyAwareCNNRPMRegressor(nn.Module):

    def __init__(
        self,
        dropout=0.20,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Preserve frequency;
            # downsample time only.
            nn.MaxPool2d(
                kernel_size=(1, 2)
            ),


            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Again preserve frequency.
            nn.MaxPool2d(
                kernel_size=(1, 2)
            ),


            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Only one frequency reduction.
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            # Preserve 32 frequency positions;
            # collapse time.
            nn.AdaptiveAvgPool2d(
                (32, 1)
            ),
        )


        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 32,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        x = self.regressor(
            x
        )

        return x.squeeze(-1)
