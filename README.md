# SatelliteVision 🛰️
### AI-Powered Satellite Image Classification

SatelliteVision is a web-based platform developed as part of my Computer Science and Technology thesis at Wuhan University of Technology. It uses deep learning to classify satellite imagery into five land-cover categories and provides an interface for exploring classification results.

## Project Overview

The project investigates the use of pre-trained deep learning models for satellite image classification in geographically held-out areas.

The classification system focuses on five land-cover classes:

1. **Tree Cover**
2. **Grassland**
3. **Cropland**
4. **Built-up**
5. **Permanent Water**

The platform combines a web interface with a Python-based backend to support image processing and model inference.

## Research Context

**Thesis title:** Improving Fine-Tuning Efficiency of Pre-trained Deep Learning Models for Five-Class Satellite Image Classification in Geographically Held-Out Areas.

**Institution:** Wuhan University of Technology  
**Degree:** BSc in Computer Science and Technology  
**Author:** Ilyass Zelmimi  
**Year:** 2026

## Technology Stack

- **Python** – Backend and machine learning development
- **PyTorch** – Deep learning model development
- **ResNet18** – Image classification architecture
- **TypeScript** – Frontend development
- **React** – User interface
- **Vite** – Frontend development and build tooling
- **Rasterio / NumPy** – Geospatial and numerical data processing

## Project Structure

```text
satellitevision/
├── backend/          # Python backend
├── public/           # Static assets
├── samples/          # Sample resources
├── segmentation/     # Image processing components
├── src/              # Frontend source code
├── index.html
├── package.json
├── vite.config.ts
└── README.md
```

## Key Features

- Web-based satellite image analysis interface
- Five-class land-cover classification
- Deep learning-based image inference
- Geospatial image processing support
- Interactive project demonstration

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ilyassdvf/satellitevision.git
cd satellitevision
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Start the frontend

```bash
npm run dev
```

### 4. Start the backend

Open a separate terminal, navigate to the backend directory, and follow the backend dependency instructions.

The backend entry point is `backend/main.py`. Confirm its required Python packages and startup command before running it.

## Research Focus

The associated research explores how fine-tuning strategies and labelled-data budgets affect classification performance and adaptation cost when models are applied to geographically unseen satellite imagery.

The broader objective is to investigate the balance between:

- Classification performance
- Number of labelled training patches
- Trainable model parameters
- Training time

## Current Scope and Limitations

- The project focuses on five land-cover categories.
- The current model performs patch-level classification; it should not be interpreted as full-resolution pixel-wise segmentation.
- Supported image formats and input requirements depend on the current backend implementation.
- Model performance should be evaluated using documented experimental results rather than assumed from the interface.

## Author

**Ilyass Zelmimi**  
Computer Science and Technology  
Wuhan University of Technology

GitHub: [@ilyassdvf](https://github.com/ilyassdvf)

---

*SatelliteVision — Applying deep learning to understand our changing Earth.*