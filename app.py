from kivi_cache import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try different models — just change the name, same API
MODEL = "facebook/opt-125m"  

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
text = generate(
    model, tokenizer,
    "In the future, artificial intelligence will",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    verbose=True,
)
print(text)