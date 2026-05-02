# 🩻 Radiology Image Captioning using BLIP

## Overview
This project builds an AI-based system that generates captions for radiology images, specifically **chest X-rays**, using a transformer-based vision-language model.

It leverages the **BLIP (Bootstrapped Language Image Pretraining)** model to learn mappings between medical images and textual descriptions.

---

## Key Features
- Uses **ROCO (Radiology Objects in COntext)** dataset
- Filters dataset to focus on **chest X-ray images**
- Applies **image captioning using BLIP**
- Supports GPU acceleration (via PyTorch)
- Mixed precision training using `torch.cuda.amp`

---

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Datasets (Hugging Face)
- Albumentations (for preprocessing)
- Kaggle API (dataset download)
- Matplotlib (visualization)

---

## Dataset
- **ROCO Dataset** (Radiology Images + Captions)
- Source: Kaggle  
- Automatically downloaded using Kaggle API

### Dataset Processing
- Loads CSV metadata
- Filters captions containing **"chest x-ray"**
- Maps image paths for training

---

## Model Used
- `BLIP (BlipForConditionalGeneration)`
- Pretrained transformer model for image captioning

---

## Project Workflow

1. Install dependencies
2. Upload Kaggle API key (`kaggle.json`)
3. Download and extract dataset
4. Load and filter dataset (chest X-rays only)
5. Preprocess images and captions
6. Train BLIP model
7. Generate captions for medical images

---

## Installation

```bash
pip install datasets==2.16.1 albumentations kaggle gradio
pip install transformers>=4.36 torch torchvision
```

---

## Setup Instructions

### 1. Upload Kaggle API Key
```python
from google.colab import files
files.upload()
```

### 2. Configure Kaggle
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset
The notebook automatically downloads:

ROCO Dataset → /content/roco

---

## Directory Structure

```
/content
 ├── roco/
 │    ├── all_data/
 │    │    ├── train/
 │    │    │    ├── radiology/images/
 │    │    │    └── radiologytraindata.csv
 │
 ├── working/train/
 ├── kaggle.json
 └── notebook.ipynb
```

---

## Example Use Case

Input: Chest X-ray image  
Output: Generated medical caption describing abnormalities or findings

---

## Applications
- Clinical decision support
- Automated radiology reporting
- Medical AI research
- Assistive tools for radiologists

---

## Limitations
- Dataset is filtered only by keyword ("chest x-ray")
- No domain-specific fine-tuning beyond dataset filtering
- Requires GPU for efficient training

---

## Future Improvements
- Fine-tune on specialized medical datasets (e.g., MIMIC-CXR)
- Add evaluation metrics (BLEU, ROUGE)
- Build a web interface using Gradio
- Deploy as an API service

---

## Author
Final Year B.Tech Project (AI/ML Focus)

---

## License
This project is for academic and research purposes.
