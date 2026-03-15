# RAG Policy Assistant

> End-to-end **RAG over policy/regulatory documents** with hybrid retrieval (FAISS + BM25), optional LoRA fine-tuning, 4-bit quantization, and RAGAS evaluation. Built for CPU or limited GPU (consumer hardware). Ready for portfolio / senior-engineer showcase.

**Tech stack:** LangChain, FastAPI, sentence-transformers, FAISS, BM25, PEFT/LoRA, RAGAS, BitsAndBytes.

---

## Architecture Overview

### Hybrid Search (Ensemble Retrieval)

The system uses **LangChain's EnsembleRetriever** to combine:

| Component | Role | Weight |
|-----------|------|--------|
| **FAISS** (vector store) | Semantic search — finds by *meaning* | 50% |
| **BM25** | Lexical/keyword search — finds *exact terms* (e.g. policy IDs, dates) | 50% |

This ensures answers reflect both **conceptual relevance** and **keyword precision** (e.g. "Section 4.2", "effective date", "pre-existing condition").

- **Chunking**: `RecursiveCharacterTextSplitter` — chunk size **1000**, overlap **200** to preserve context across segments.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly).
- **Generation**: `google/flan-t5-small` by default; optional 4-bit quantized Llama/Phi for larger models on limited hardware.

### LoRA Fine-Tuning

A standalone script (`scripts/finetune_lora.py`) demonstrates **Low-Rank Adaptation (LoRA)** for instruction-tuning on policy documents:

**Note:** LoRA training and using a LoRA adapter at inference need more RAM and benefit from a GPU. On a typical PC without a strong GPU, use the **default generator (no LoRA)**—the rest of the RAG pipeline (retrieval, API, Streamlit) runs fine on CPU.

- **Stack**: `peft` + `transformers` + `bitsandbytes` (QLoRA for 4-bit base).
- **Target models**: Llama-3-8B, Phi-3, or similar.
- **Use case**: Adapt the LLM to your policy/regulatory wording and citation style via a small set of (question, context, answer) triples.

Run from project root:

```bash
python -m scripts.finetune_lora --model_name microsoft/Phi-3-mini-4k-instruct --output_dir ./artifacts/lora_policy
```

---

## Prerequisites

- **Python 3.10+**
- This repo includes **sample policy documents** in `data/` (health benefits, leave policy, code of conduct) so you can run the agent immediately. Add your own PDFs, `.txt`, or `.md` to `data/` as needed, then rebuild the index.

## Quick Start

```bash
# Clone (or download) and enter project
cd rag-policy-assistant

# Install dependencies
pip install -r requirements.txt

# Build index (ingests data/ with RecursiveCharacterTextSplitter, builds FAISS + BM25)
python -m scripts.build_index

# Serve API (RAG chain + /query)
uvicorn app.main:app --reload --port 8000
# Open http://127.0.0.1:8000/docs

# Optional: Streamlit UI (point to API)
streamlit run src/ui_streamlit.py
```

---

## Project Layout

```
├── app/
│   └── main.py              # FastAPI app: /query (answer + source documents)
├── src/
│   ├── ingest.py            # RecursiveCharacterTextSplitter ingestion
│   ├── retrieval.py         # HybridRetriever (FAISS + BM25, 50/50)
│   ├── generate.py          # Generator + 4-bit quantization support
│   ├── reranker.py          # Cross-encoder reranker
│   └── ui_streamlit.py      # Streamlit front-end
├── scripts/
│   ├── build_index.py       # Build hybrid index from data/
│   ├── finetune_lora.py     # LoRA fine-tuning (Llama/Phi + instruction template)
│   ├── test_lora_feature.py # Smoke tests for default generator and optional LoRA
│   └── eval_ragas.py        # RAGAS: Faithfulness, Answer Relevancy, Context Precision
├── data/                    # PDF, .txt, .md policy documents
├── artifacts/
│   └── index/               # FAISS index + meta.json
├── requirements.txt
└── README.md
```

---

## API: `/query`

**POST** `/query` (or **POST** `/ask` for backward compatibility)

- **Request**: `{ "query": "What is the waiting period for pre-existing conditions?", "top_k": 5 }`
- **Response**: `{ "answer": "...", "source_documents": [{ "content": "...", "metadata": { "source", "page" } }], "latency_ms": 120 }`

---

## Evaluation (RAGAS)

Run the evaluation pipeline with sample questions and optional ground-truth answers:

```bash
python -m scripts.eval_ragas --questions_file eval/sample_questions.json --output eval/ragas_results.json
```

Metrics:

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Is the retrieved context precise (minimal irrelevant content)?

---

## Using a LoRA adapter at inference

After training with `scripts/finetune_lora.py`, use the saved adapter in the API. (Requires enough RAM/GPU to load the 4-bit base model—see LoRA note above.)

1. Set **`LORA_PATH`** to the adapter directory (e.g. `./artifacts/lora_policy` or `./artifacts/lora_policy_test`).
2. Optionally set **`GENERATION_MODEL`** to the same base model you fine-tuned (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `microsoft/Phi-3-mini-4k-instruct`). If unset, the app defaults to Phi-3 when `LORA_PATH` is set.
3. Start the API; the generator will load the base model in 4-bit and apply the PEFT adapter.

```bash
set LORA_PATH=./artifacts/lora_policy_test
set GENERATION_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
uvicorn app.main:app --reload --port 8000
```

Then call **POST /query** as usual; answers will use the fine-tuned adapter.

---

## Quantization (4-bit)

To run a larger model (e.g. Llama-3-8B) on limited GPU/RAM, the generator supports **BitsAndBytesConfig** 4-bit quantization. Set environment or config:

- `USE_4BIT=true` and `GENERATION_MODEL=meta-llama/Llama-3-8B` (or similar) to load in 4-bit.

---

## License

MIT — see [LICENSE](LICENSE).

## References


- LangChain [EnsembleRetriever](https://python.langchain.com/docs/modules/model_io/prompts/ensemble_retriever), [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_character_text_splitter)
- RAGAS: [explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- PEFT/LoRA: [huggingface/peft](https://github.com/huggingface/peft)
