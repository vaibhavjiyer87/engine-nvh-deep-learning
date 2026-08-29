# Project Working-Style Retrospective

## 1. Engineering-first model selection

The project consistently treated machine learning as an engineering
tool rather than assuming that a more complex architecture must
produce a better result.

Strong conventional baselines were created before neural-network
development began.

Model decisions remained tied to frozen quantitative criteria.


## 2. Explicit reproducible workflows

Development was most effective when each activity contained:

- a numeric step identifier
- an exact file path
- runnable code
- an expected result
- a clear stopping condition

This structure made multi-session Colab development substantially
easier to manage.


## 3. Strong artifact discipline

The project persisted:

- configurations
- manifests
- split definitions
- feature fingerprints
- model checkpoints
- normalization records
- experiment tables
- decision reports
- final figures

Large raw data and model binaries were kept outside GitHub.


## 4. Test-set discipline

Training data was used for fitting.

Validation data was used for model development and selection.

Rejected CNN and MTL candidates were not evaluated on the test set.

This prevented the test split from becoming an additional tuning
dataset.


## 5. Time-boxed development

The project was completed across multiple focused sessions.

The most reliable closeout procedure was:

1. persist generated artifacts
2. save the notebook
3. commit and synchronize GitHub
4. record the last completed step
5. define the next resume point


## 6. Controlled hypothesis testing

Later deep-learning experiments were not arbitrary architecture
changes.

Each tested a specific hypothesis:

- CNN-002: alternative regression loss
- CNN-003: preserve absolute frequency information
- MTL-001: joint RPM and torque supervision
- MTL-002: hybrid learned and engineered feature representation

When the final hypothesis failed to outperform the benchmarks, model
development stopped.


## Process improvements for future projects

### Check compute resources immediately before training

GPU availability should be confirmed immediately before any
long-running training cell.

### Persist recovery state from Epoch 1

Checkpoint, optimizer, early-stopping and training-shuffle state
should be saved after every epoch from the beginning.

### Create a master experiment registry earlier

Future projects should maintain one canonical experiment table from
the first model onward rather than consolidating it only at project
closeout.


## Preferred future workflow

Freeze specification

→ run one controlled experiment

→ persist evidence

→ make a decision

→ stop or version the next hypothesis


This project benefited more from disciplined experiment design,
artifact management and model-selection rigor than it would have
from simply testing a larger number of neural-network variants.
