import os, re, torch
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER","0")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import runpod

BASE_MODEL   = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "dickyybayu/mistral-recommender-finetune-id")
HF_TOKEN     = os.getenv("HF_TOKEN")
USE_4BIT     = os.getenv("USE_4BIT", "1") == "1"
OFFLOAD_DIR  = os.getenv("OFFLOAD_DIR", "offload")

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
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=dtype, quantization_config=q, offload_folder=OFFLOAD_DIR, **auth)
    m = PeftModel.from_pretrained(base, LORA_ADAPTER, **auth)
    t = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, **auth)
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    t.padding_side = "left"
    gc = GenerationConfig(do_sample=False, max_new_tokens=128, pad_token_id=t.eos_token_id)
    model, tokenizer, gen_cfg = m, t, gc

def create_product_prompt(name: str, price: int, stock: int, viewers: str, event: str):
    return f"""Kamu adalah host live TikTok Shop yang berbahasa santai. Berdasarkan detail produk di bawah, buat 1 kalimat promosi singkat, hype, dan persuasif agar penonton segera beli.

Contoh jawaban:
COPY: Sandal kece anti licin wajib punya buat gaya santai kamu!
HOST: Rini
TIME: 18:00-20:00 WIB
BUNDLE: Sandal Jepit Stylish Anti Licin + Topi Keren (diskon 15%)

Nama Produk   : {name}
Harga (diskon): Rp{price:,}
Stok Tersisa  : {stock}
Penonton Live : {viewers}
Event         : {event}"""

def generate_once(text: str, do_sample: bool, temperature: float, top_p: float, max_new_tokens: int):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=top_p if do_sample else None, max_new_tokens=max_new_tokens, pad_token_id=gen_cfg.pad_token_id)
    return clean_response(tokenizer.decode(out[0], skip_special_tokens=True))

def handler(event):
    load()
    ip = event.get("input", {}) or {}
    system_prompt = ip.get("system_prompt", "Kamu adalah host live TikTok Shop yang berbahasa santai. Jawab PERSIS 4 baris dengan label COPY:, HOST:, TIME:, BUNDLE:. Tanpa tambahan kalimat lain.")
    max_new_tokens = int(ip.get("max_new_tokens", gen_cfg.max_new_tokens))
    do_sample = str(ip.get("do_sample", "true")).lower() in ("1","true","t","yes","y")
    temperature = float(ip.get("temperature", 0.7))
    top_p = float(ip.get("top_p", 0.9))
    prompt = ip.get("prompt")

    if prompt:
        text = wrap_inst(prompt, system_prompt) if system_prompt else prompt
        out = generate_once(text, do_sample, temperature, top_p, max_new_tokens)
        return {"output": out}

    name = ip.get("name")
    price = ip.get("price")
    stock = ip.get("stock")
    viewers = ip.get("viewers") or "ribuan"
    event_name = ip.get("event")
    num_variations = max(1, int(ip.get("num_variations", 3)))

    missing = [k for k, v in {"name": name, "price": price, "stock": stock, "event": event_name}.items() if v in (None, "")]
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}

    try:
        price = int(price)
        stock = int(stock)
    except Exception:
        return {"error": "price and stock must be integers"}

    product_prompt = create_product_prompt(name, price, stock, viewers, event_name)
    wrapped = wrap_inst(product_prompt, system_prompt) if system_prompt else product_prompt
    variations = [generate_once(wrapped, do_sample, temperature, top_p, max_new_tokens) for _ in range(num_variations)]
    return {"variations": variations}

runpod.serverless.start({"handler": handler})
