# 🏦 Banking77 Chatbot with Gemma-2B

[![Streamlit Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-demo-link.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A fine-tuned chatbot for banking queries, using LoRA-adapted Gemma-2B on the Banking77 dataset.

<img src="https://i.imgur.com/JfnX7Wg.gif" width="600" alt="Chatbot Demo">

## Features

- **Precision Responses**: 77 banking intent templates
- **Efficient Fine-Tuning**: LoRA adapters (~100MB)
- **Streamlit UI**: Local deployment ready
- **Kaggle Compatible**: Optimized for T4 GPUs

## Quick Start

### 1. Local Deployment
```bash
git clone https://github.com/yourusername/banking-chatbot.git
cd banking-chatbot
pip install -r requirements.txt
streamlit run app.py
```

### 2. Using the Chatbot
```python
from chatbot import BankingBot

bot = BankingBot()
response = bot.ask("How do I block my card?")
print(response)  # "We’re sorry to hear that. Please block via..."
```

## Project Structure
```
banking-chatbot/
├── app.py                # Streamlit interface
├── banking-chatbot-output/  # LoRA adapters
├── training.ipynb       # Kaggle training notebook
├── banking_instructions.csv  # Processed dataset
└── requirements.txt
```

## Training Details

| Hyperparameter       | Value       |
|----------------------|-------------|
| Base Model           | Gemma-1.1-2B-it |
| Epochs               | 2           |
| LoRA Rank (r)        | 4           |
| Batch Size           | 1           |
| Learning Rate        | 2e-5        |

## Customization

Edit `response_templates.py` to add new banking responses:
```python
response_templates = {
    "new_intent": "Custom response text...",
    # ...
}
```

## Limitations

- Requires GPU for inference
- Accuracy ~65% on unseen queries
- English only

## Requirements

⚠ **GPU Required**  
For standard operation with `google/gemma-1.1-2b-it`, an NVIDIA GPU with ≥8GB VRAM is needed.

For CPU-only systems, modify `app.py`:  
```python
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # CPU-compatible