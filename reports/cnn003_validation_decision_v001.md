# CNN-003 Validation Decision — v001

## Experiment

CNN-003 introduced a frequency-aware CNN architecture intended to
preserve absolute frequency information relevant to engine-order
movement with RPM.

The existing data population, split, log-mel representation and
frozen primary selection metric were retained.

## Training recovery

The original training runtime lost its Google Drive transport
connection after completing Epoch 22. Training was
resumed from Epoch 23 using the persisted model,
optimizer and early-stopping state. No completed epoch was repeated.

The frozen HDF5 feature cache was mirrored to local Colab storage
for the remaining training and evaluation operations.

## Best result

Best epoch: 19

Validation MAE: 62.73 RPM

BASE-002 validation MAE: 37.62 RPM

## Decision

CNN-003 did not improve upon BASE-002 on the frozen primary validation MAE metric. It was therefore not approved for test evaluation.

Decision status: `not_selected_for_test`
