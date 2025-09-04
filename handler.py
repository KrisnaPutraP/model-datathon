import os, re, torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import runpod

BASE_MODEL   = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "dickyybayu/mistral-recommender-finetune-id")
HF_TOKEN     = os.getenv("HF_TOKEN")
USE_4BIT     = os.getenv("USE_4BIT", "1") == "1"

model = None
tokenizer = None
gen_cfg = None

def wrap_inst(user_prompt: str, system_prompt: str):
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"

def clean_response(text: str):
    return re.sub(r"(</?s>|</?INST>)", "", text, flags=re.IGNORECASE).strip()

def load():
    global model, tokenizer, gen_cfg
    if model is not None:
        return
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4") if USE_4BIT else None
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=dtype, quantization_config=q, offload_folder="offload", **auth)
    model = PeftModel.from_pretrained(base, LORA_ADAPTER, **auth)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, **auth)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

def handler(event):
    load()
    ip = event.get("input", {})
    prompt = ip.get("prompt", "")
    system_prompt = ip.get("system_prompt", "Kamu adalah host live TikTok Shop yang berbahasa santai. Jawab PERSIS 4 baris dengan label COPY:, HOST:, TIME:, BUNDLE:. Tanpa tambahan kalimat lain.")
    max_new_tokens = int(ip.get("max_new_tokens", 128))
    gen_cfg.max_new_tokens = max_new_tokens
    text = wrap_inst(prompt, system_prompt) if system_prompt else prompt
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=gen_cfg.max_new_tokens, do_sample=gen_cfg.do_sample, pad_token_id=gen_cfg.pad_token_id)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"output": clean_response(raw)}

runpod.serverless.start({"handler": handler})
