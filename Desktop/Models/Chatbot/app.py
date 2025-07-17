import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
MODEL_DIR = "./banking-chatbot-output"
BASE_MODEL = "google/gemma-1.1-2b-it"  # Switch to "TinyLlama-1.1B" if crashes persist
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Micro Templates ---
RESPONSE_TEMPLATES = {
    "balance": "Balances: Dial *247#",
    "card": "Card block: Call 0700-XXXXXX",
    "loan": "Loans: 0800-LOAN-123",
    "transfer": "Transfers: Use mobile app"
}

# --- Memory-Efficient Loading ---
@st.cache_resource
def load_model():
    try:
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"⚠️ Insufficient memory. Try: 1) Close other apps 2) Restart PC")
        return None

# --- Streamlit UI ---
def main():
    st.title("🏦 Banking Mini-Assistant")
    
    if st.button("Load Model (Click Only When Ready)"):
        with st.spinner("Loading (may take 3-5 mins)..."):
            model = load_model()
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
        if model:
            query = st.text_input("Ask about balances, cards, etc:")
            if query:
                response = RESPONSE_TEMPLATES.get(query.lower(), "Try: 'balance', 'card', 'loan', or 'transfer'")
                st.write(f"Assistant: {response}")

if __name__ == "__main__":
    main()