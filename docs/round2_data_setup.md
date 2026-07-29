# Round 2 Data Setup

## What changed

The archive `data/VAI_NVS_DATA_ROUND2.zip` was extracted into:

```text
data/raw/phase2/
```

The archive contains scene directories directly at its root rather than an
additional split directory. The extracted layout is therefore:

```text
data/raw/phase2/<scene>/train/images/
data/raw/phase2/<scene>/train/sparse/0/
data/raw/phase2/<scene>/test/test_poses.csv
```

The following scenes are available:

```text
HCM0421
HCM0539
HCM0540
HCM0644
HCM0674
bonsai
chair
```

## Compatibility path

The phase runner expects a split component in its path:

```text
data/raw/<phase>/<split>/<scene>/...
```

To support the existing runner without copying the 1.2 GB dataset, the
following symlink was added:

```text
data/raw/phase2/round2 -> .
```

This makes both of these paths resolve to the same scene data:

```text
data/raw/phase2/HCM0421/
data/raw/phase2/round2/HCM0421/
```

## Validation

All seven scenes were checked for:

- training images;
- COLMAP sparse data under `train/sparse/0/`;
- test pose files under `test/test_poses.csv`.

The archive does not contain test images, which is expected because test
ground truth is withheld.

## Running the pipeline

Use `round2` as the split name and `phase2` as the phase:

```bash
make phase-run SCENE=HCM0421 SPLIT=round2 PHASE=phase2
```

Replace `HCM0421` with any of the other scene names. The raw data is ignored
by Git, so the extracted dataset and symlink do not create source-control
changes.
