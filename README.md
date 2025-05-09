# Bird Sound Classification using CNNs

This project classifies bird species based on audio recordings by converting them into spectrogram images and training convolutional neural networks (CNNs). 
The workflow includes data preparation, image generation, and model training with cross-validation.

---

## 📁 Files Overview

### `EDA_final.ipynb` – Data Preprocessing & Spectrogram Generation
- Loads audio metadata and raw `.ogg` files
- Trims or pads audio clips to 15 seconds
- Applies a **BioDenoise** model to remove background noise
- Converts each audio clip into a **Mel spectrogram**
- Normalizes and saves spectrograms as `.png` files in the `Local/` directory
- Outputs a new `metadata.csv` mapping image files to bird labels

> ✅ Use this notebook to generate the cleaned spectrogram dataset before training

---

### `Training.ipynb` – Model Training (Single Run)
- Loads the `metadata.csv` and spectrogram images
- Uses **PyTorch** and **EfficientNet-B0**
- Trains a CNN model on the spectrogram images
- Splits the data into training and validation sets (80/20 split)
- Tracks loss and accuracy during training
- Optionally plots training history and confusion matrix

> ✅ Ideal for testing model performance or debugging before full cross-validation

---

### `train_cnn_10fold.py` – 10-Fold Cross-Validation Training
- Loads `metadata.csv` and filters out rare species with fewer than 10 samples
- Performs **stratified 10-fold cross-validation**
- Uses **EfficientNetV2-S** pretrained on ImageNet
- Trains the model and reports **fold-by-fold accuracy**
- Returns the **mean accuracy** across all folds

> ✅ Best for final model evaluation and fair performance estimation

---

## 📂 Directory Structure
├── metadata.csv # File-to-label mapping
├── EDA_final.ipynb # Preprocessing and spectrogram generation
├── Training.ipynb # Baseline training
├── train_cnn_10fold.py # Cross-validation training
