import torch
import torch.nn as nn


class MultiTaskRPMTorqueCNN(nn.Module):

    def __init__(
        self,
        dropout=0.20,
    ):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),

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

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.AdaptiveAvgPool2d(
                (32, 1)
            ),
        )


        self.shared_projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 32,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),
        )


        self.rpm_head = nn.Linear(
            64,
            1,
        )


        self.torque_head = nn.Linear(
            64,
            1,
        )


    def forward(
        self,
        x,
    ):

        x = self.encoder(
            x
        )

        shared = self.shared_projection(
            x
        )


        rpm = self.rpm_head(
            shared
        ).squeeze(-1)


        torque = self.torque_head(
            shared
        ).squeeze(-1)


        return rpm, torque
