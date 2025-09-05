import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import runpod

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LORA_ADAPTER = os.getenv("LORA_ADAPTER", "dickyybayu/mistral-recommender-finetune-v2")
HF_TOKEN = os.getenv("HF_TOKEN")
USE_4BIT = os.getenv("USE_4BIT", "1") == "1"
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "offload")

model = None
tokenizer = None
gen_cfg = None

def wrap_inst(user_prompt: str, system_prompt: str | None = None):
    if system_prompt is None or system_prompt == "":
        return f"<s>[INST] {user_prompt} [/INST]"
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt} [/INST]"

def clean_response(text: str):
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    text = re.sub(r"</?s>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[/?INST\]", "", text, flags=re.IGNORECASE)
    return text.strip()

def create_three_product_prompt(p1, p2, p3):
    template = """Kamu adalah host live TikTok Shop yang berbahasa santai. WAJIB buat copywriting BARU dan SPESIFIK untuk masing-masing produk di bawah ini. JANGAN copy template atau contoh apapun!

Tugas:
1. Buat copywriting unik untuk SETIAP produk yang disebutkan
2. Pilih 2 produk termahal untuk bundle dengan diskon
3. Tentukan jam live yang optimal

FORMAT OUTPUT:
COPY1: [copywriting khusus untuk produk 1]
COPY2: [copywriting khusus untuk produk 2]
COPY3: [copywriting khusus untuk produk 3]
BUNDLE: [nama produk A] + [nama produk B] (diskon X%)
TIME: [jam:menit-jam:menit] WIB

PRODUK YANG HARUS DIBUATKAN COPYWRITING:

PRODUK 1:
Nama: {name1}
Harga: Rp{price1:,}
Terjual hari ini: {sold1} pcs

PRODUK 2:
Nama: {name2}
Harga: Rp{price2:,}
Terjual hari ini: {sold2} pcs

PRODUK 3:
Nama: {name3}
Harga: Rp{price3:,}
Terjual hari ini: {sold3} pcs

BUAT COPYWRITING SESUAI PRODUK DI ATAS, BUKAN PRODUK LAIN!"""
    return template.format(
        name1=p1["name"], price1=int(p1["price"]), sold1=int(p1["sold"]),
        name2=p2["name"], price2=int(p2["price"]), sold2=int(p2["sold"]),
        name3=p3["name"], price3=int(p3["price"]), sold3=int(p3["sold"]),
    )

def parse_response(response: str):
    parsed = {"copy1": "", "copy2": "", "copy3": "", "bundle": "", "time": ""}
    for line in response.split("\n"):
        s = line.strip()
        if s.startswith("COPY1:"):
            parsed["copy1"] = s[6:].strip()
        elif s.startswith("COPY2:"):
            parsed["copy2"] = s[6:].strip()
        elif s.startswith("COPY3:"):
            parsed["copy3"] = s[6:].strip()
        elif s.startswith("BUNDLE:"):
            parsed["bundle"] = s[7:].strip()
        elif s.startswith("TIME:"):
            parsed["time"] = s[5:].strip()
    return parsed

def load():
    global model, tokenizer, gen_cfg
    if model is not None:
        return
    auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    q = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4"
    ) if USE_4BIT else None
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=q,
        offload_folder=OFFLOAD_DIR,
        **auth
    )
    m = PeftModel.from_pretrained(base, LORA_ADAPTER, **auth)
    m.eval()
    t = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, **auth)
    if t.pad_token_id is None:
        t.pad_token_id = t.eos_token_id
    t.padding_side = "left"
    gc = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        max_new_tokens=400,
        pad_token_id=t.eos_token_id
    )
    model, tokenizer, gen_cfg = m, t, gc

def generate_once(text: str, do_sample: bool = None, temperature: float = None, top_p: float = None, max_new_tokens: int = None, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(int(seed))
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    use_sample = gen_cfg.do_sample if do_sample is None else bool(do_sample)
    temp = gen_cfg.temperature if temperature is None else float(temperature)
    topp = 0.9 if top_p is None else float(top_p)
    mnt = gen_cfg.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
    kwargs = {"do_sample": use_sample, "max_new_tokens": mnt, "pad_token_id": gen_cfg.pad_token_id}
    if use_sample:
        kwargs.update({"temperature": temp, "top_p": topp})
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    return clean_response(tokenizer.decode(out[0], skip_special_tokens=False))

def handler(event):
    load()
    ip = event.get("input", {}) or {}
    system_prompt = ip.get("system_prompt", "")
    prompt = ip.get("prompt")
    products = ip.get("products")
    do_sample = ip.get("do_sample")
    temperature = ip.get("temperature")
    top_p = ip.get("top_p")
    max_new_tokens = ip.get("max_new_tokens")
    seed = ip.get("seed")

    if prompt:
        text = wrap_inst(prompt, system_prompt)
        out = generate_once(text, do_sample, temperature, top_p, max_new_tokens, seed)
        parsed = parse_response(out)
        filled = sum(1 for v in parsed.values() if v.strip())
        score = (filled / 5) * 100
        return {"output": out, "parsed": parsed, "quality_score": score}

    if isinstance(products, list) and len(products) == 3:
        for p in products:
            if not all(k in p for k in ("name", "price", "sold")):
                return {"error": "Each product must have name, price, sold"}
        product_prompt = create_three_product_prompt(products[0], products[1], products[2])
        wrapped = wrap_inst(product_prompt, system_prompt)
        out = generate_once(wrapped, do_sample, temperature, top_p, max_new_tokens, seed)
        parsed = parse_response(out)
        filled = sum(1 for v in parsed.values() if v.strip())
        score = (filled / 5) * 100
        return {"output": out, "parsed": parsed, "quality_score": score}

    return {"error": "Provide either 'prompt' or 'products' (list of exactly 3 items with name, price, sold)"}

runpod.serverless.start({"handler": handler})
