
# Zero-Shot Anomaly Detection using Vision Foundation Models

This repository contains the code, experiments, and documentation for the DTU 02456 Deep Learning project **"Zero-Shot Anomaly Detection using Vision Foundation Models"** (Group 26).

---

# Getting Started — Running the Setup

To prepare the environment, install dependencies, and configure the dataset automatically, use the provided setup script for your operating system.

## Windows

Run the batch setup script:

```powershell
.\setup.bat
```

This script will:

- Create or reuse a Python virtual environment (`.venv`)
- Optionally install all Python dependencies from `requirements.txt`
- Detect GPU / CUDA availability
- Optionally download and extract the **MVTec AD dataset**

## Linux / macOS

Make the setup script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
```

The Linux version provides the same functionality:

- Virtual environment creation and activation
- Optional dependency installation
- GPU detection (CUDA)
- Optional dataset download + extraction

## After Setup

Once the setup script completes, you can:

- Run the pipeline using the mainpipeline.ipynb

---

## Running the Pipeline

There are two main components:
1. The main pipeline produces both pooled and patch embeddings. Note that the embeddings are saved to .npz files which are then loaded into the pipeline. This reduces computational time for iteration, removing the need for the creation of new embeddings each time. Graphical results for PCA, anomaly maps, and F1 Optuna studies are printed in the notebook directly.
2. The GMM study loads in the pooled embeddings created by the main pipeline. This file is separate to reduce clutter, and optimizes paremeters to obtain the F1 score for classification using mean-pooled embeddings.


## Project Overview

Industrial quality inspection often suffers from a lack of defective samples and the unpredictable nature of anomalies. Traditional supervised deep learning methods require large labeled datasets of both normal and defective samples, which are costly and impractical to obtain in industrial settings.

This project explores **zero-shot anomaly detection** using **pre-trained vision foundation models** such as **DINOv3**, **Mirroring DINO**, and **Segment Anything (SAM)**. The goal is to design a **generalizable anomaly detection pipeline** that can identify surface defects without any additional training — reducing the need for labeled data and retraining.

## Objectives

- Apply large pre-trained **vision models** (e.g., DINOv3, SAM) to detect anomalies in industrial images.
- Leverage **patch-level embeddings** to compare normal and defective regions.
- Develop a **zero-shot detection pipeline** evaluated on the **MVTec Anomaly Detection (AD)** dataset.
- Optionally extend the project with **few-shot classification** or **prompt-based feature adaptation**.

## Background

Recent progress in **self-supervised vision models** such as DINOv3 and SAM has shown strong generalization capabilities across diverse tasks. These models learn transferable, high-level visual features that can describe textures and shapes without supervision.

- **MVTec AD Dataset** provides a benchmark for industrial anomaly detection.
- **Flow Matching for Unsupervised Anomaly Detection** demonstrates modeling normal data distributions using flow-based methods.
- **CLIP for Few-Shot Inspection** highlights how foundation models can be adapted for manufacturing visual inspection tasks.

This project builds upon these insights to develop a **zero-shot anomaly localization and detection pipeline** using pre-trained foundation model embeddings.

## Project Milestones

| Week | Milestone |
|------|------------|
| 9 | Literature review on zero-shot anomaly detection and setup of the MVTec AD dataset. Initialize GitHub and Overleaf. |
| 10 | Implement the anomaly detection pipeline using DINOv3 and experiment with hyperparameter settings. |
| 11 | Review first results. If time allows, start developing anomaly classification extensions. |
| 12 | Debug and clean the pipeline; start drafting the report. |
| 13 | Finalize and submit the report. |

## Technologies

- **Frameworks:** PyTorch, NumPy, OpenCV, Matplotlib  
- **Models:** DINOv3, Mirroring DINO, SAM  
- **Dataset:** MVTec AD

## References

1. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR.  
2. Oquab, M., Caron, M., Assran, M., Misra, I., & Joulin, A. (2025). *DINOv3: Scaling Self-Supervised Learning for Universal Vision Representations*. arXiv:2508.10104.  
3. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). *Segment Anything*. arXiv:2304.02643.  
4. Li, L., Han, C., Lu, Z., Gao, J., Fang, J., & Zheng, Z. (2025). *How and Why: Taming Flow Matching for Unsupervised Anomaly Detection and Localization*. arXiv:2508.05461.  
5. Megahed, F. M., Chen, Z., & Lo, S. (2025). *Adapting OpenAI’s CLIP Model for Few-Shot Image Inspection in Manufacturing Quality Control*. arXiv:2501.12596.

---

### Contributors
- D. Lam (s252126)  
- W. Wang (s251983)  
- A. Pedraza (s251894)

**DTU 02456 Deep Learning — Group 26**  
November 2025
