<p align="center">
  <h1 align="center">EB-FaLCom</h1>
  <p align="center"><b>Error-Bounded Federated Learning Compressor with Gradient-Aware Prediction</b></p>
</p>

<p align="center">
  <a href="https://github.com/apecs-lab/EB-FaLCom">
    <img src="https://img.shields.io/badge/APPFL-Extension-blue" alt="APPFL Extension">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

---

## 📖 Overview

**EB-FaLCom** is a gradient-aware, predictor-enhanced error-bounded lossy compressor designed for federated learning (FL). This repository extends the [APPFL framework](https://github.com/APPFL/APPFL)[...]

- **Temporal smoothness** across training rounds
- **Layer-wise and kernel-wise structural consistency** in gradients
- **Two-level bitmap encoding** for predictable patterns

The result? **Higher compression ratios while preserving training accuracy** in federated learning scenarios.

### Key Features

✨ **Gradient-aware magnitude predictor** - Exploits temporal correlation across FL rounds  
🔍 **Sign predictor** - Supports oscillation-based and kernel-level consistency  
📦 **Compact encoding** - Two-level bitmap for predictable kernels and dominant signs  
🔧 **EBLC-compatible pipeline** - Quantizer, entropy coding, and lossless compression following SZ3 design principles  

---

## 📁 Project Structure

```
EB-FaLCom/
├── src/appfl/
│   └── compressor/
│       └── FalCom.py              # Main compressor implementation
├── examples/
│   ├── resources/
│   │   └── configs/
│   │       └── {dataset_name}/    # Configuration files for each dataset
│   │           ├── server_fedavg.yaml   # Default config with SZ3/SZ2/ZFP
│   │           └── server_FaLCom.yaml   # FaLCom-specific config
│   └── run_exp.sh                 # Runnable FL experiment script
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv

### Step 1: Create a Virtual Environment

We **highly recommend** using a clean Conda environment:

```bash
conda create -n falcom python=3.8
conda activate falcom
```

Alternatively, using virtualenv:

```bash
python -m venv falcom-env
source falcom-env/bin/activate  # On Windows: falcom-env\Scripts\activate
```

### Step 2: Install APPFL

Install APPFL with examples support (MPI optional):

```bash
pip install pip --upgrade
pip install "appfl[examples,mpi]"
```

💡 **Note**: If you don't need MPI for simulations, install without the `mpi` option:
```bash
pip install "appfl[examples]"
```

#### Ubuntu Users

If installation fails due to MPI dependencies:

```bash
sudo apt install libopenmpi-dev libopenmpi-bin libopenmpi-doc
```

### Step 3: Clone This Repository

```bash
git clone https://github.com/apecs-lab/EB-FaLCom.git
cd EB-FaLCom
```

---

## 🚀 Quick Start

### Run the Example Experiment

A ready-to-use script is provided in `examples/run_exp.sh`:

```bash
bash examples/run_exp.sh
```

This script:
1. Starts an APPFL federated learning experiment
2. Applies the FalCom compressor (`src/compressor/FalCom.py`) during training
3. Outputs compression ratios and training metrics

---

## ⚙️ Configuration

### Configuration Files

Configuration files are located in `examples/resources/configs/{dataset_name}/` directory. You can choose between:

#### 1. **server_fedavg.yaml** - Default Configuration with Built-in Compressors

Use this configuration to experiment with APPFL's built-in compressors (SZ3, SZ2, ZFP). Modify the `lossy_compressor` field in the configuration:

```yaml
client_configs:
  comm_configs:
    compressor_configs:
      enable_compression: True
      lossy_compressor: "SZ3"  # Options: "SZ3", "SZ2", "ZFP"
      lossless_compressor: "blosc"
      sz_config:
        error_bounding_mode: "REL"  # "REL" or "ABS"
        error_bound: 1e-3
```

#### 2. **server_FaLCom.yaml** - FaLCom-Specific Configuration

Use this configuration to enable **FaLCom** with customizable hyperparameters:

```yaml
client_configs:
  train_configs:
    # Enable gradient transmission (required for FaLCom)
    send_gradient: True
    num_local_epochs: 1
    optim: "SGD"
    optim_args:
      lr: 0.1
      momentum: 0.9
      weight_decay: 5e-4

  comm_configs:
    compressor_configs:
      enable_compression: True
      lossy_compressor: "FaLCom"
      lossless_compressor: "blosc"
      param_cutoff: 1024
      
      # FaLCom-specific hyperparameters
      momentum_lr: 0.07              # Learning rate for momentum-based magnitude prediction
      consistency_threshold: 0.5     # Threshold for sign consistency prediction
      
      sz_config:
        error_bounding_mode: "REL"   # "REL" (relative) or "ABS" (absolute)
        error_bound: 1e-3            # Error bound value

server_configs:
  num_global_epochs: 10              # More epochs benefit temporal prediction
  aggregator: "FedAvgAggregator"
  aggregator_kwargs:
    expect_gradient: True            # Must be True for FaLCom
```


### Hyperparameter Tuning

#### For Built-in Compressors (SZ3/SZ2/ZFP)

Key parameters to adjust in `server_fedavg.yaml`:

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `lossy_compressor` | Compressor type | "SZ3", "SZ2", "ZFP" |
| `error_bound` | Maximum allowed error | 1e-4 to 1e-2 |
| `error_bounding_mode` | "REL" (relative) or "ABS" (absolute) | - |
| `param_cutoff` | Minimum parameter size for compression | 512-2048 |

#### For FaLCom

Key parameters to adjust in `server_FaLCom.yaml`:

| Parameter | Description | Recommended Range | Impact |
|-----------|-------------|-------------------|--------|
| `error_bound` | Maximum allowed error | 1e-4 to 1e-2 | Higher = more compression, lower accuracy |
| `error_bounding_mode` | Error bound type | "REL" or "ABS" | REL adapts to gradient magnitude |
| `momentum_lr` | Learning rate for magnitude predictor | 0.05-0.3 | Higher = faster adaptation |
| `consistency_threshold` | Sign prediction confidence threshold | 0.3-0.7 | Lower = more aggressive prediction |
| `param_cutoff` | Minimum parameter size to compress | 512-2048 | Avoid compressing small layers |

**Important Notes:**
- `send_gradient: True` is **required** for FaLCom to work
- `expect_gradient: True` must be set in aggregator config
- Higher `momentum_lr` makes the magnitude predictor adapt faster but may be less stable
- Lower `consistency_threshold` enables more aggressive sign prediction but may reduce accuracy


## 🧩 Compressor Design

### Core Components

The FalCom compressor (`src/compressor/FalCom.py`) consists of:

#### 1. **Gradient-Aware Magnitude Predictor**
- Exploits temporal correlation between consecutive FL rounds
- Uses momentum-based prediction with configurable learning rate (`momentum_lr`)
- Predicts gradient magnitudes based on historical patterns
- Reduces residual data size significantly

#### 2. **Sign Predictor**
- **Oscillation-based prediction**: Tracks sign flips across rounds
- **Kernel-level consistency**: Exploits structural patterns within layers
- Uses `consistency_threshold` to determine prediction confidence
- Achieves high sign prediction accuracy

#### 3. **Two-Level Bitmap Encoding**
- **First level**: Encodes predictable vs. unpredictable kernels
- **Second level**: Encodes dominant signs within unpredictable kernels
- Compact representation reduces metadata overhead

#### 4. **EBLC-Compatible Pipeline**
- Quantizer for error-bounded compression (SZ3-style)
- Entropy coding for residual data
- Lossless compression (Blosc)
- Ensures improvements are attributable to predictor enhancements

### Compression Workflow

```
Gradients → Magnitude Predictor (momentum_lr) → Sign Predictor (consistency_threshold)
    ↓                                                      ↓
Residuals → Quantizer (error_bound) → Entropy Coder → Lossless Compressor (blosc)
                                                      ↓
                                              Compressed Data
```

---

## 📊 Performance

FalCom achieves:
- **Up to 53% higher compression ratios** compared to baseline EBLC methods
- **Negligible accuracy loss** in FL training
- **Reduced communication overhead** in distributed FL scenarios

---

## 📚 Documentation

For detailed information about APPFL:
- [APPFL Documentation](http://appfl.rtfd.io/)
- [APPFL GitHub](https://github.com/APPFL/APPFL)

---

<p align="center">
  Made with ❤️ by the APECS Lab Team
</p>
