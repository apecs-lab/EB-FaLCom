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

**EB-FaLCom** is a gradient-aware, predictor-enhanced error-bounded lossy compressor designed for federated learning (FL). This repository extends the [APPFL framework](https://github.com/APPFL/APPFL) with advanced compression techniques that leverage:

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
├── src/
│   └── compressor/
│       └── FalCom.py          # Main compressor implementation
├── examples/
│   └── run_exp.sh             # Runnable FL experiment script
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
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

### Step 4: (Optional) Install Additional Dependencies

If your project requires additional packages, install them:

```bash
pip install -r requirements.txt  # If you have a requirements.txt
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


## 🧩 Compressor Design

### Core Components

The FalCom compressor (`src/compressor/FalCom.py`) consists of:

#### 1. **Gradient-Aware Magnitude Predictor**
- Exploits temporal correlation between consecutive FL rounds
- Predicts gradient magnitudes based on historical patterns
- Reduces residual data size significantly

#### 2. **Sign Predictor**
- **Oscillation-based prediction**: Tracks sign flips across rounds
- **Kernel-level consistency**: Exploits structural patterns within layers
- Achieves high sign prediction accuracy

#### 3. **Two-Level Bitmap Encoding**
- **First level**: Encodes predictable vs. unpredictable kernels
- **Second level**: Encodes dominant signs within unpredictable kernels
- Compact representation reduces metadata overhead

#### 4. **EBLC-Compatible Pipeline**
- Quantizer for error-bounded compression
- Entropy coding for residual data
- Lossless compression (compatible with SZ3 design)
- Ensures improvements are attributable to predictor enhancements

### Compression Workflow

```
Gradients → Magnitude Predictor → Sign Predictor → Two-Level Bitmap
    ↓                                                      ↓
Residuals → Quantizer → Entropy Coder → Lossless Compressor
    ↓
Compressed Data
```

---

## 📊 Performance

FalCom achieves:
- **Up to 53% higher compression ratios** compared to baseline EBLC methods
- **Negligible accuracy loss** in FL training
- **Reduced communication overhead** in distributed FL scenarios



## 📚 Documentation

For detailed information about APPFL:
- [APPFL Documentation](http://appfl.rtfd.io/)
- [APPFL GitHub](https://github.com/APPFL/APPFL)


<p align="center">
  Made with ❤️ by the APECS Lab Team
</p>
