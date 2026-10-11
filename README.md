#  My Gen-Z Chadbot

A personality-driven chatbot that talks in Gen-Z slang. It is built by fine-tuning **Llama-3 8B Instruct** with **LoRA** on a small custom dataset, then served through a **FastAPI** backend with a browser chat UI and an optional **web-search** mode. Everything runs on a **free Google Colab T4 GPU**.

> **Chad** is confident, funny, and keeps answers short, helpful, and full of vibe.

---

##  Features

- **Custom personality** via LoRA fine-tuning on hand-written (user → reply) pairs
- **4-bit quantization (NF4)** so an 8B model fits on a free T4 (16 GB)
- **Loss masking**: the model learns only from Chad's replies, not from the user prompts
- **FastAPI server** with `/chat` and `/search` endpoints
- **Single-page chat UI** (dark theme, conversation history, typing indicator)
- **Optional web search** using the [Serper](https://serper.dev) API to ground answers in live results
- **Public link** through ngrok, so you can share the bot from Colab
- Adapter pushed to the Hugging Face Hub, so the app never needs to retrain

---

##  Tech Stack

| Layer | Tools |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Fine-tuning | Hugging Face `transformers`, `peft` (LoRA), `bitsandbytes`, `accelerate`, `datasets` |
| Serving | FastAPI, Uvicorn, pyngrok |
| Search (optional) | Serper (Google Search API) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Compute | Google Colab (T4 GPU) |

---

## Project Structure

```
.
├── my_genz_chadbot_finetune.ipynb   # Train the LoRA adapter and push it to the Hub
├── my_genz_chadbot_app.ipynb        # Load the adapter and launch the chat app
├── train_data.jsonl                 # (optional) training pairs, if split out of the notebook
├── index.html                       # (optional) chat page, if split out of the notebook
└── README.md
```

Currently the training pairs live inside the fine-tune notebook and the chat page is embedded in the app notebook. You can move them into `train_data.jsonl` and `index.html` as the project grows.

---

##  Quick Start

### Prerequisites

1. A Google account (for Colab)
2. A [Hugging Face](https://huggingface.co) account with:
   - **Access approved** for [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
   - A **Write** access token
3. A free [ngrok](https://ngrok.com) auth token (for the app notebook)
4. *(Optional)* A [Serper](https://serper.dev) API key for the web-search toggle

### Step 1: Fine-tune (`my_genz_chadbot_finetune.ipynb`)

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU**.
2. Run the cells in order:
   1. Install dependencies
   2. Log in to Hugging Face (paste your Write token)
   3. Edit or extend the `pairs` list (aim for 100+ examples)
   4. Load Llama-3 in 4-bit
   5. Convert the data to Llama-3 chat format
   6. Attach LoRA adapters
   7. Train
   8. Test with a few prompts
   9. Set `HF_USERNAME` to your own username and push the adapter to the Hub

### Step 2: Run the app (`my_genz_chadbot_app.ipynb`)

1. Open the notebook in Colab with a **T4 GPU** runtime.
2. In the model-loading cell, set `ADAPTER_ID` to your own repo (e.g. `your-username/my-genz-chadbot`).
3. Run the cells in order. Provide your ngrok token when asked. Press Enter to skip the Serper key if you don't want search.
4. Click the printed `YOUR CHADBOT IS LIVE: https://...` link and start chatting.

> ngrok may show a "Visit Site" interstitial the first time. Just click through.

**Tip:** Store `NGROK_AUTH_TOKEN` and `SERPER_API_KEY` in Colab **Secrets** (🔑 icon) and the notebook will pick them up automatically instead of prompting.

---

##  How It Works

### Training

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quant, fp16 compute (T4 has no good bf16 support) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Batch size | 1 (gradient accumulation of 4) |
| Learning rate | 2e-4 |
| Epochs | 5 for ~30 examples; use ~3 once you pass 100 |
| Optimizer | `paged_adamw_8bit` with gradient checkpointing |
| Max sequence length | 512 tokens |

Each example is formatted with the Llama-3 chat template (system prompt, user turn, assistant turn). The prompt tokens are masked with `-100`, so the loss is computed only on Chad's reply.

### Serving

- The base model is loaded in 4-bit and the LoRA adapter is attached with `PeftModel`.
- Generation uses `temperature=0.7`, `top_p=0.9`, and stops on both the standard EOS token and `<|eot_id|>`.
- A thread lock serializes generation calls, so concurrent requests don't collide on the single GPU.
- The last 6 messages of history are sent with each request to keep context short.

### API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/chat` | Plain chat with the fine-tuned model |
| `POST` | `/search` | Fetches the top 5 Serper snippets and answers using them as context |

**Request body (both POST routes):**

```json
{
  "message": "what does rizz mean?",
  "history": [
    { "role": "user", "content": "hey" },
    { "role": "assistant", "content": "Yooo what's good!" }
  ]
}
```

**Response:**

```json
{ "response": "Rizz is your charm, how smooth you are with people. 😎" }
```

---

## Customizing Chad

- **Change the personality:** edit `SYSTEM_PROMPT` and the `pairs` list. Use the same `SYSTEM_PROMPT` in both notebooks.
- **Improve quality:** more and more varied examples help the most. If the bot parrots training lines word for word, lower `EPOCHS`.
- **Restyle the UI:** edit the `HTML_PAGE` string in the app notebook (or move it into `index.html`).
- **Longer answers:** raise `max_new_tokens` in `generate_reply` (200 for chat, 250 for search by default).

---

## Limitations

- **Tiny dataset (~30 pairs):** the style transfers well, but factual accuracy comes from the base model, and it can hallucinate.
- **Search mode is simple:** it feeds raw search snippets into the prompt and does not cite sources.
- **Colab is temporary:** the ngrok link dies when the session ends, and the free tier has GPU time limits.
- **One request at a time:** generation is locked, so this is a demo server, not a production one.
- **Gated model:** Llama-3 requires accepting Meta's license on Hugging Face.

---

## Roadmap

- [ ] Grow the dataset to 100+ diverse examples
- [ ] Move training data to `train_data.jsonl` and the UI to `index.html`
- [ ] Streaming responses in the UI
- [ ] Source citations for web-search answers
- [ ] Permanent hosting (e.g. Hugging Face Spaces) instead of Colab + ngrok
- [ ] Evaluation: compare against the base model on a held-out prompt set

---

##  Acknowledgements

- [Meta Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [FastAPI](https://fastapi.tiangolo.com), [ngrok](https://ngrok.com), [Serper](https://serper.dev)

---

## License

Add your license here (e.g. MIT). Note that use of the base model is governed by the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).

---

**Author:** Mohit · [GitHub](https://github.com/mohitsharmas97)
