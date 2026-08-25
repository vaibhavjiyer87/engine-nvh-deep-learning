# MTL-001 Validation Decision — v001

## Model

MTL-001 uses one shared frequency-aware acoustic encoder with
separate RPM and torque regression heads.

## Selection protocol

Checkpoint selection used joint normalized validation MAE.

Final test approval requires:

1. MTL-001 must not be worse than the frozen shared-population
   conventional benchmark on either RPM MAE or torque MAE.
2. MTL-001 must improve at least one of the two targets.

## Validation results

### RPM

BASE-002 shared-population MAE:
38.39 RPM

MTL-001 MAE:
113.44 RPM

Change:
+195.52%

### Torque

TBASE-002 shared-population MAE:
23.32 Nm

MTL-001 MAE:
36.07 Nm

Change:
+54.67%

## Decision

`not_selected_for_test`

The test split has not influenced model training or validation
selection.
