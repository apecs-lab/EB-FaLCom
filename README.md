🌀 FedGAZ (FalCom): Gradient-Aware Error-Bounded Lossy Compressor for Federated Learning

This repository extends APPFL
 with a gradient-aware, predictor-enhanced error-bounded lossy compressor for federated learning (FL).
The compressor leverages temporal smoothness and layer-wise/kernel-wise structural consistency in gradients to achieve higher compression ratios while preserving training accuracy.
The core implementation is in:

src/compressor/FalCom.py

⚙️ Installation

We recommend using a clean Conda environment.

conda create -n falcom python=3.8
conda activate falcom


Install APPFL (with examples; MPI optional):

pip install "appfl[examples,mpi]"
# or, without MPI:
# pip install "appfl[examples]"


Clone this repository:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

🚀 Run the Example

A runnable script is provided in examples/run_exp.sh:

bash examples/run_exp.sh


This script starts an APPFL experiment and uses our compressor in
src/compressor/FalCom.py.

🧩 What’s Inside the Compressor

Gradient-aware magnitude predictor (uses temporal correlation across rounds)

Sign predictor (supports oscillation-based and kernel-level consistency)

Two-level bitmap to compactly encode predictable kernels and dominant signs

EBLC-compatible pipeline: quantizer, entropy coding, and lossless compression follow the same design as SZ3, so that improvements can be attributed to our predictor.

📁 Project Layout (minimal)
.
├── src/
│   └── compressor/
│       └── FalCom.py        # main compressor
├── examples/
│   └── run_exp.sh           # example script to run FL + FalCom
└── README.md
