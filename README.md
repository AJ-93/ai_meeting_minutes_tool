# AI Meeting Minutes Tool

A Google Colab notebook that turns a meeting audio recording into structured minutes using open-source models, run locally on the Colab GPU. It uses **4-bit quantization (NF4 via `bitsandbytes`)** to run the LLM in a fraction of the memory.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/<repo>/blob/main/ai_meeting_minutes_tool.ipynb)

## How it works

1. **Transcribe:** `openai/whisper-large-v3-turbo` converts the audio file to text.
2. **Summarize:** `meta-llama/Llama-3.2-3B-Instruct` (4-bit quantized) writes the minutes in Markdown.

Output includes a summary (attendees, location, date), discussion points, takeaways, and action items with owners.

## Quantization with bitsandbytes

The LLM is loaded with a `BitsAndBytesConfig` so it fits comfortably on a T4 GPU:

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, device_map="auto", quantization_config=quant_config
)
```

| Setting | What it does |
|---|---|
| `load_in_4bit=True` | Stores weights in 4 bits instead of 16, cutting weight memory roughly 4x. |
| `bnb_4bit_quant_type="nf4"` | NormalFloat4: a 4-bit data type designed for normally distributed weights, giving better accuracy than plain int4 or fp4. |
| `bnb_4bit_use_double_quant=True` | Quantizes the quantization constants too, saving additional memory. |
| `bnb_4bit_compute_dtype=torch.bfloat16` | Weights are stored in 4-bit but dequantized to bf16 for matrix math, preserving output quality. |


## Requirements

- Google Colab with a **GPU runtime** (T4 or better). 4-bit quantization via `bitsandbytes` needs CUDA and does not run on Mac.
- A Hugging Face account with access to the [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) license accepted.
- An `HF_TOKEN` stored in Colab Secrets (key icon in the sidebar).
- An audio file in Google Drive.

## Usage

1. Open the notebook in Colab and select `Runtime > Change runtime type > GPU`.
2. Add `HF_TOKEN` to Colab Secrets and enable notebook access.
3. Set `audio_file` to your audio path in Drive.
4. Run all cells. The minutes render at the end of the notebook.
