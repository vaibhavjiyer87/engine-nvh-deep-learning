import torch
import torch.nn as nn


class CNNRPMRegressor(nn.Module):

    def __init__(
        self,
        dropout=0.20,
    ):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
        )


        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                32,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
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

        return x.squeeze(
            -1
        )
