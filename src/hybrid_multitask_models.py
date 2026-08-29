import torch
import torch.nn as nn


class HybridRPMTorqueMTL(nn.Module):

    def __init__(
        self,
        engineered_feature_count=41,
        dropout=0.20,
    ):

        super().__init__()


        # ----------------------------------------------------
        # Acoustic branch
        # Same frequency-aware encoder family used in MTL-001.
        # ----------------------------------------------------

        self.acoustic_encoder = nn.Sequential(

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


        self.acoustic_projection = nn.Sequential(

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


        # ----------------------------------------------------
        # Engineered NVH feature branch
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(

            nn.Linear(
                engineered_feature_count,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                32,
            ),

            nn.ReLU(),
        )


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                64 + 32,
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
        logmel,
        engineered_features,
    ):

        acoustic = (
            self.acoustic_encoder(
                logmel
            )
        )

        acoustic = (
            self.acoustic_projection(
                acoustic
            )
        )


        engineered = (
            self.feature_projection(
                engineered_features
            )
        )


        fused = torch.cat(
            [
                acoustic,
                engineered,
            ],
            dim=1,
        )


        shared = self.fusion(
            fused
        )


        rpm = (
            self.rpm_head(
                shared
            )
            .squeeze(-1)
        )


        torque = (
            self.torque_head(
                shared
            )
            .squeeze(-1)
        )


        return rpm, torque
