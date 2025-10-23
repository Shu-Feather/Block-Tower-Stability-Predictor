# Lab 3: Block Tower Stability Predictor

## 📋 Project Overview

This project implements an InceptionV4-based deep learning model for predicting the stability of block towers. Using the ShapeStacks dataset, the model performs binary classification to determine whether a block tower will collapse (stable vs. unstable).

### Background

Predicting structural stability is a critical challenge in architecture, robotics, and disaster prevention. Humans can quickly judge stability through visual intuition and experience—a skill rooted in implicit physical knowledge and causal reasoning. This project aims to replicate this capability using deep learning, bridging the gap between visual perception and physical reasoning.

### Key Features

- **InceptionV4 Architecture**: State-of-the-art CNN for image classification
- **Binary Classification**: Predicts whether a tower is stable (0) or unstable (1)
- **ShapeStacks Dataset**: Synthetic block tower images with stability annotations
- **Data Augmentation**: Extensive augmentation pipeline for robust training
- **TensorBoard Integration**: Real-time training monitoring and visualization

---

## 🗂️ Project Structure

```
lab3/
├── data_provider/
│   └── shapestacks_provider.py      # Data loading and preprocessing
├── tf_models/
│   └── inception/
│       ├── inception_v4.py          # InceptionV4 architecture
│       └── inception_model.py       # Model definition and training logic
├── intuitive_physics/
│   └── stability_predictor/
│       ├── train_inception_v4_shapestacks.py   # Training script
│       └── test_inception_v4_shapestacks.py    # Testing script
├── lab3_dataset/                    # This should be downloaded mannually
│   └── shapestacks/                 # Dataset directory
│       ├── recordings/              # Image data
│       ├── splits/                  # Train/eval/test splits
│       │   └── ccs_all/
│       │       ├── train.txt
│       │       ├── eval.txt
│       │       ├── test.txt
│       │       └── *_bgr_mean.npy
│       ├── mjcf/                    # MuJoCo simulation files
│       └── meta/                    # Metadata and blacklists
├── output_ccs_all/                  # Training output directory
│   ├── snapshots/                   # Best model checkpoints
│   └── events.out.tfevents.*        # TensorBoard logs
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- Python 3.6+
- CUDA 10.0+ (for GPU support)
- cuDNN 7.0+

### Environment Setup

```bash
# Create conda environment
conda create -n stability_predictor python=3.6
conda activate stability_predictor

conda install cudatoolkit=10.0
conda install cudnn==7.6.5

# Install TensorFlow (GPU version)
pip install tensorflow-gpu==1.15.0

# Install other dependencies
pip install numpy matplotlib Pillow
```

---

## 📦 Dataset

### Download

Download the ShapeStacks dataset from the provided link and extract it to `lab3_dataset/shapestacks/`.

### Dataset Structure

```
shapestacks/
├── recordings/          # RGB images of block towers
│   └── env_*/
│       ├── rgb-*.png
│       ├── log.txt      # Contains "Stack collapse: True/False"
│       └── depth_log.txt
├── splits/
│   └── ccs_all/         # Data split configuration
│       ├── train.txt    # Training scenarios
│       ├── eval.txt     # Validation scenarios
│       ├── test.txt     # Test scenarios
│       └── *_bgr_mean.npy  # Mean values for normalization
└── mjcf/                # MuJoCo configuration files
```

### Label Convention

- **Label 0**: Stable tower (Stack collapse: False)
- **Label 1**: Unstable tower (Stack collapse: True)

---

## 🚀 Quick Start

### 1. Set Environment Variable

```bash
export SHAPESTACKS_CODE_HOME=./
```

### 2. Training

```bash
# Basic training (recommended settings)
export CUDA_VISIBLE_DEVICES=0  # Use single GPU, you can modify it

python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /path/to/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --train_epochs 40 \
  --batch_size 16 \
  --memcap 0.6

# Run in background with logging
nohup python intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py \
  --data_dir /path/to/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --train_epochs 40 \
  --batch_size 16 > train.log 2>&1 &
```

### 3. Monitor Training

```bash
# Monitor log file
tail -f train.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Start TensorBoard
tensorboard --logdir=./output_ccs_all --port=6006
# Open browser: http://localhost:6006
```

### 4. Testing

```bash
python intuitive_physics/stability_predictor/test_inception_v4_shapestacks.py \
  --data_dir /path/to/lab3_dataset/shapestacks \
  --model_dir ./output_ccs_all \
  --split_name ccs_all \
  --batch_size 32
```

---

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Path to ShapeStacks dataset |
| `--model_dir` | Required | Directory for model outputs |
| `--split_name` | `ccs_all` | Dataset split to use |
| `--train_epochs` | `40` | Total training epochs |
| `--epochs_per_eval` | `1` | Evaluation frequency |
| `--batch_size` | `32` | Training batch size |
| `--n_best_eval` | `5` | Number of best checkpoints to keep |
| `--memcap` | `0.8` | GPU memory fraction to use |
| `--n_prefetch` | `32` | Number of batches to prefetch |

### Data Augmentation

Default augmentations applied during training:
- Random cropping and resizing
- Random horizontal flip
- Random rotation (±2 degrees)
- Color jittering (brightness, saturation, hue, contrast)
- Gaussian noise
- Value clipping

```bash
# Customize augmentation
python train_inception_v4_shapestacks.py \
  --augment crop flip recolour clip \
  ...
```

---

## 📊 Expected Results

### Training Progress

```
Epoch 1/40
INFO:tensorflow:loss = 0.693147, step = 1
INFO:tensorflow:loss = 0.682340, step = 100
...

Evaluation results:
  accuracy: 0.652341
  global_step: 5000
  loss: 0.621234

Top-5 models so far:
  Accuracy: 0.652341 - eval=0.652341
```

### Performance Metrics

| Metric | Expected Value |
|--------|---------------|
| Training Accuracy | > 85% |
| Validation Accuracy | > 80% |
| Test Accuracy | > 75% |
| Training Time (40 epochs) | ~6-12 hours (on A100) |

### GPU Memory Usage

| Batch Size | GPU Memory | A100 80GB Status |
|------------|------------|------------------|
| 32 | ~40-50 GB | ⚠️ May OOM |
| 16 | ~25-30 GB | ✅ Recommended |
| 8 | ~15-20 GB | ✅ Safe |
| 4 | ~10-15 GB | ✅ Very Safe |

---

## 📈 Monitoring and Visualization

### TensorBoard Metrics

Available metrics:
- **Loss**: Training and validation loss curves
- **Accuracy**: Training and validation accuracy
- **Learning Rate**: RMSProp learning rate schedule
- **Images**: Sample input images (if `--display_inputs > 0`)

### Output Files

```
output_ccs_all/
├── model.ckpt-*              # Latest checkpoint
├── checkpoint                # Checkpoint metadata
├── events.out.tfevents.*     # TensorBoard logs
├── graph.pbtxt               # Model graph
├── snapshots/                # Best models
│   ├── eval=0.850000/
│   │   ├── model.ckpt-*
│   │   └── checkpoint
│   └── ...
└── info_*.txt                # Training configuration
```

---

## 🔬 Advanced Usage

### Custom Learning Rate Schedule

Modify `inception_model.py`:

```python
# Add learning rate decay
global_step = tf.train.get_global_step()
learning_rate = tf.train.exponential_decay(
    learning_rate=_START_LEARNING_RATE,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.train.RMSPropOptimizer(
    learning_rate=learning_rate,
    decay=_DECAY,
    epsilon=_EPSILON
)

# Log learning rate
tf.summary.scalar('learning_rate', learning_rate)
```

### Multi-GPU Training

```bash
# Use multiple GPUs (experimental)
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_inception_v4_shapestacks.py \
  --batch_size 64 \
  ...
```

### Resume Training

Training automatically resumes from the latest checkpoint if `model_dir` contains previous checkpoints:

```bash
# Will resume from existing checkpoint
python train_inception_v4_shapestacks.py \
  --model_dir ./output_ccs_all \
  ...
```

---

## 📚 References

1. **InceptionV4 Paper**: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
2. **ShapeStacks Dataset**: [ShapeStacks: Learning Vision-Based Physical Intuition for Generalised Object Stacking](https://arxiv.org/abs/1804.08018)
3. **TensorFlow Estimator API**: [TensorFlow Estimator Guide](https://www.tensorflow.org/guide/estimator)

---

## 🤗 Contact

If you find this project insightful and helpful, or if you find there exist bugs in the project, feel free to email:
shuyoung@stanford.edu