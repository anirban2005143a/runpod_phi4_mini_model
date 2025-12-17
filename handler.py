import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_ID = "priyankrathore/phi4Mini_PEFT_layman"
CACHE_DIR = "/runpod-volume/hf-cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["HF_HOME"] = CACHE_DIR

# -------------------------------------------------
# LOAD MODEL WITH CACHE CHECK + LOGGING
# -------------------------------------------------
def load_model_with_cache():
    try:
        print("🔍 Checking Hugging Face cache...")
        try:
            snapshot_path = snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
            print(f"✅ Using cached model at: {snapshot_path}")
        except (FileNotFoundError, HFValidationError):
            print("⬇️ Model not found in cache. Downloading now...")
            snapshot_path = snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=CACHE_DIR,
                local_files_only=False
            )
            print(f"✅ Model downloaded and cached at: {snapshot_path}")

        print("🚀 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True, cache_dir=CACHE_DIR)

        print("🚀 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(snapshot_path, trust_remote_code=True)
        model.to(DEVICE)
        model.eval()

        print(f"🎯 Model ready on {DEVICE}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        raise

# -------------------------------------------------
# GLOBAL LOAD
# -------------------------------------------------
MODEL, TOKENIZER = load_model_with_cache()

# -------------------------------------------------
# CHUNKING FUNCTION
# -------------------------------------------------
def chunk_text(text, tokenizer, max_tokens=4000, overlap_tokens=50):
    separators = ["\n\n", "\n", ". "]
    def token_length_function(chunk):
        return len(tokenizer(chunk)["input_ids"])
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_length_function,
        is_separator_regex=False
    )
    return splitter.split_text(text)

# -------------------------------------------------
# TRIM FUNCTION
# -------------------------------------------------
def trim_to_last_fullstop(text: str) -> str:
    last_dot_index = text.rfind(".")
    if last_dot_index != -1:
        return text[:last_dot_index+1]
    return text

# -------------------------------------------------
# SUMMARY GENERATION
# -------------------------------------------------
def generate_summary(model, tokenizer, text, device, params):
    chunks = chunk_text(text, tokenizer, max_tokens=params["chunk_tokens"], overlap_tokens=params.get("chunk_overlap",50))
    summaries = []

    for i, chunk in enumerate(chunks,1):
        print(f"   🔸 Summarizing chunk {i}/{len(chunks)} ...")
        prompt = (
            "Summarize the following legal text in simple layman terms. "
            "Focus only on the main issue, what the petitioner claimed, "
            "what the government replied, and what the court decided.\n\n"
            f"{chunk}"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params.get("do_sample", False),
                temperature=params.get("temperature",0.0),
                pad_token_id=tokenizer.eos_token_id
            )

        summary = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        summary = trim_to_last_fullstop(summary)
        summaries.append(summary)

    return {"generated_text": "\n\n".join(summaries)}

# -------------------------------------------------
# RUNPOD HANDLER
# -------------------------------------------------
def handler(job):
    print(f"Received job: {job['id']}")

    if not MODEL or not TOKENIZER:
        return {"error": "Model not loaded."}

    try:
        job_input = job["input"]
        case_text = job_input["text"]
    except KeyError:
        return {"error": "Missing required field: 'text'"}

    if not case_text.strip():
        return {"error": "Input text cannot be empty."}

    generation_params = {
        "max_new_tokens": job_input.get("max_new_tokens", 1024),
        "chunk_tokens": job_input.get("chunk_tokens", 2000),
        "chunk_overlap": job_input.get("chunk_overlap", 100),
        "do_sample": job_input.get("do_sample", False),
        "temperature": job_input.get("temperature", 0.0)
    }

    try:
        summary = generate_summary(MODEL, TOKENIZER, case_text, DEVICE, generation_params)
        return {
            "summary": summary,
            "input_text_length": len(case_text),
            "parameters_used": generation_params
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
