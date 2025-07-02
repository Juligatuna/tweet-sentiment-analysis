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
 