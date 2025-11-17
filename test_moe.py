from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer

base_model = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram/hf_cache/models--unsloth--meta-llama-3.1-8b-instruct-bnb-4bit/snapshots/f15c379fb32bb402fa06a7ae9aecb1febf4b79ec"
adapter_dir = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram/hf_cache/science_model"
tok_dir = "/home/hice1/rswaminathan38/scratch/CS-6200-Project_ram/hf_cache/science_tokenizer"

model, _ = FastLanguageModel.from_pretrained(
    model_name=base_model,
    dtype=torch.float16,
    load_in_4bit=False,
    trust_remote_code=True,
    local_files_only=True,
)
model.load_adapter(adapter_dir)
model = FastLanguageModel.for_inference(model)
tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)

inp = tok(["What is gravity?"], return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
