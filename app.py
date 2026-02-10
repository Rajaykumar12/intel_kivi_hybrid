from kivi_cache import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)
print(text)