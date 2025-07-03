# Tweet Sentiment Analysis Dashboard

This repository contains a sentiment analysis project using the DistilBERT model fine-tuned on the Sentiment140 dataset. It includes:

- Data cleaning and preprocessing
- Sentiment prediction using a TensorFlow DistilBERT model
- Visualizations and exploratory data analysis
- A Streamlit web app (`App.py`) for interactive sentiment predictions and visualizations

## Project Structure

- `App.py` — Streamlit dashboard for EDA, predictions, and performance
- `distilbert_sentiment/` — Saved model and tokenizer files
- `notebook.ipynb` — Jupyter notebook with data processing and model training
- `data/` — Folder for datasets (if included)
- `requirements.txt` — Required Python packages

## Getting Started

### Prerequisites

- Python 3.7+
- Install dependencies:
  ```bash
  pip install -r requirements.txt

### Running the Streamlit App
To launch the app, run:

streamlit run App.py

The app will open in your default browser.

### Dataset
This project uses the Sentiment140 dataset.

### Model
A DistilBERT model fine-tuned on Sentiment140 for binary sentiment classification (positive/negative).
 

### 📥 Download Pre-trained Model
Due to GitHub’s 100MB file size limit, the pre-trained model (tf_model.h5) is not stored directly in this repository.

Instead, download it from Google Drive: 
https://drive.google.com/drive/folders/1XQkQDYAlrAxw3V_zJ7mw6hy72UnuHmWw?usp=drive_link

### 👉 Download the model from Google Drive folder

After downloading, place tf_model.h5 inside the distilbert_sentiment/ folder.

### 🔄 Optional: Download Model Automatically in Code
You can also download the model in your Python code using gdown:

import gdown

url = "https://drive.google.com/uc?export=download&id=FILE_ID"
output = "distilbert_sentiment/tf_model.h5"
gdown.download(url, output, quiet=False)

Install gdown:

pip install gdown
