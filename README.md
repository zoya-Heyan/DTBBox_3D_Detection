# DT-BBox 3D Detection

3D object detection with DT-BBox (Dual Target Bounding Box) approach.

## Project Structure

```
DTBBox_3D_Detection/
│
├── configs/              # Configuration files
│   └── kitti.yaml
│
├── data/                 # Dataset
│   └── kitti/            # KITTI dataset
│       ├── training/     # Training data
│       │   ├── image_2/  # Images
│       │   ├── label_2/  # Labels
│       │   └── calib/    # Calibration files
│       └── ImageSets/    # Split files
│           ├── train.txt
│           └── val.txt
│
├── datasets/             # Dataset loaders
│   ├── __init__.py
│   └── kitti_dataset.py
│
├── models/               # Model definitions
│   ├── __init__.py
│   ├── backbone.py
│   └── dtbbox_net.py
│
├── modules/              # Network modules
│   ├── __init__.py
│   ├── pair_graph.py
│   ├── roi_utils.py
│   └── rpfo.py
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── box_ops.py
│   ├── geometry.py
│   ├── losses.py
│   ├── metrics.py
│   ├── misc.py
│   └── visualize.py
│
├── checkpoints/          # Model checkpoints
├── outputs/              # Output results
│
├── train.py              # Training script
├── eval.py               # Evaluation script
├── demo.py               # Demo script
└── README.md             # This file
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install torch torchvision opencv-python numpy pyyaml
   ```

2. **Prepare KITTI dataset**:
   - Download KITTI dataset from [official website](http://www.cvlibs.net/datasets/kitti/)
   - Extract to `data/kitti/` directory
   - Create split files in `data/kitti/ImageSets/`

## Training

```bash
# Train baseline model
python train.py --stage baseline

# Train dtbbox model
python train.py --stage dtbbox

# Train relative model
python train.py --stage relative

# Train full model
python train.py --stage full
```

## Evaluation

```bash
python eval.py --stage baseline --checkpoint checkpoints/checkpoint_epoch_100.pth
```

## Demo

```bash
python demo.py --stage full --checkpoint checkpoints/checkpoint_epoch_100.pth --sample_id 000001
```

## Stages

- **baseline**: Single target RoI
- **dtbbox**: Dual target RoI
- **relative**: Dual target RoI + Relative head
- **full**: Dual target RoI + Relative head + R-PFO (test time)
