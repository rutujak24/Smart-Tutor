# Mixture-of-Experts (MoE) Inference Pipeline

### CS-6200 Big Data Systems & Analytics — Project (Ramanathan Swaminathan)

This repository implements a **Mixture-of-Experts LLM inference system** combining:
- A **DistilBERT router** trained to classify input queries into semantic domains
- Four fine-tuned **LLaMA-3.1-8B Unsloth experts**:
  - `math_model` for reasoning & arithmetic
  - `science_model` for physics / chemistry / biology
  - `general_model` for everyday / commonsense queries
  - `combined_model` for code & cross-domain logic

All components run **fully offline** inside a CUDA-enabled Apptainer container on Georgia Tech’s PACE cluster.

---

## 🧱 Folder Layout

