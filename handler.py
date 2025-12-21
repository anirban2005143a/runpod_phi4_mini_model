import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_ID = "AnirbanDas2005/phi4Mini_PEFT_layman"
CACHE_DIR = "/runpod-volume/huggingface-cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["HF_HOME"] = CACHE_DIR

# -------------------------------------------------
# LOAD MODEL WITH CACHE CHECK + LOGGING
# -------------------------------------------------
def load_model_with_cache():
    try:
        print("🔍 Checking Hugging Face cache...")

        try:
            snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
            print("✅ Model found in cache")
        except Exception:
            print("⬇️ Model not found in cache. Downloading...")
            snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=CACHE_DIR,
                local_files_only=False
            )
            print("✅ Model downloaded")

        print("🚀 Loading tokenizer (slow)...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        print("🚀 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

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
# PARAGRAPH-WISE CHUNKING FUNCTION
# -------------------------------------------------
def chunk_text_paragraphwise(text, max_tokens=4000):
    """
    Split text into chunks of max_tokens using a recursive text splitter.
    First tries paragraphs (\n\n), then single lines (\n), then sentences (.).
    """
    # Initialize LangChain splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],  # hierarchical splitting
        chunk_size=max_tokens,
        chunk_overlap=0,  # no overlap
        length_function=lambda txt: len(TOKENIZER(txt, return_tensors="pt")["input_ids"][0])
    )

    chunks = splitter.split_text(text)
    return chunks

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
    chunks = chunk_text_paragraphwise(text, max_tokens=params["chunk_tokens"])
    summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"   🔸 Summarizing chunk {i}/{len(chunks)} ...")

        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following legal text in simple layman terms. "
                    "Focus only on the main issue, what the petitioner claimed, "
                    "what the government replied, and what the court decided.\n\n"
                    f"{chunk}"
                )
            }
        ]

        # Prepare prompt using chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 2048),
                do_sample=params.get("do_sample", True),
                temperature=params.get("temperature", 0.7),
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        print(f"\nSummary of chunk {i} - {response}")

        response = trim_to_last_fullstop(response)
        summaries.append(response)

    return {
        "generated_text": " \n\n ".join(summaries)
    }


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
