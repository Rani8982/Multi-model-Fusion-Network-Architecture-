# CFNet: Facial Expression Recognition via Constraint Fusion under Multi-Task Joint Learning Network

This repository contains the implementation of **CFNet**, a facial expression recognition framework based on multi-task joint learning and constraint fusion. The code is organized by dataset and implemented in **Python 3.8** with **TensorFlow 2.0**.

> 📖 Reference Paper:  
> Xiao, Junhao, et al. "CFNet: Facial expression recognition via constraint fusion under multi-task joint learning network." *Applied Soft Computing* (2023): 110312.  
> [DOI: 10.1016/j.asoc.2023.110312](https://doi.org/10.1016/j.asoc.2023.110312)

---

🔧 Environment

- Python 3.8
- TensorFlow 2.0
- NumPy, OpenCV, h5py, and other standard packages

---

 Architecture Overview

📌 Part 1: Multi-Task Joint Learning
This part trains three branches independently:
- **Global Branch**: Extracts global facial features.
- **Local Branch**: Focuses on local facial details.
- **Direct Fusion Branch**: Fuses global and local features directly.

This phase outputs three types of features:
- Global features
- Local features
- Concatenated (fused) features

📌 Part 2: Constraint Fusion
Uses the extracted features from Part 1 as input to perform adaptive fusion with constraint mechanisms based on cosine similarity and learned weights.



## 📁 Directory Structure


CFNet/
│
├── datasets/
│   ├── CK+
│
├── part1_joint_learning/
│   └── (Code for training EnhanceNet)
│
├── part2_constraint_fusion/
│   └── (Code for FusionNet using features from Part 1)
│
├── utils/
│   └── (Helper functions and preprocessing tools)
│
└── README.md
