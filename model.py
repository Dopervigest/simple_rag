import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def get_model(model_name="SweatyCrayfish/llama-3-8b-quantized"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def get_pipeline(model, tokenizer):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline
