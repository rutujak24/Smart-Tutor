from unsloth import FastLanguageModel

models = {
    "math": "KushalRamaiya/BigData_llama-3.1-8b-math",
    "science": "KushalRamaiya/BigData_llama-3.1-8b-science",
    "combined": "KushalRamaiya/BigData_llama-3.1-8b-combined",
    "general": "KushalRamaiya/BigData_llama-3.1-8b-general",
}

for key, repo in models.items():
    print(f"⏳ Downloading {key} → {repo}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=repo,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer.save_pretrained(f"./hf_cache/{key}_tokenizer")
        model.save_pretrained(f"./hf_cache/{key}_model")
        print(f"✅ Saved {key} model and tokenizer successfully.\n")
    except Exception as e:
        print(f"❌ Failed for {key}: {e}\n")
