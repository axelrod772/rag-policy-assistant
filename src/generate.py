"""
Generation with optional 4-bit quantization (BitsAndBytesConfig) for large models on limited hardware.
Supports loading a LoRA adapter (e.g. from scripts.finetune_lora) via LORA_PATH.
"""
import os
from pathlib import Path
from typing import Any

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def _get_4bit_config() -> BitsAndBytesConfig:
    """4-bit quantization for running large models on consumer hardware."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )


def get_generator(
    model_name: str | None = None,
    use_4bit: bool | None = None,
    lora_path: str | None = None,
) -> Any:
    """
    Return a text-generation pipeline.
    - Default: text2text-generation with google/flan-t5-small (no quantization).
    - If use_4bit=True (or env USE_4BIT=true): load model with BitsAndBytesConfig 4-bit
      (use with causal LM, e.g. meta-llama/Llama-3-8B, microsoft/Phi-3-mini-4k-instruct).
    - If lora_path (or env LORA_PATH) is set: load base model in 4-bit and apply PEFT adapter.
    """
    model_name = model_name or os.environ.get("GENERATION_MODEL", "google/flan-t5-small")
    use_4bit = use_4bit if use_4bit is not None else os.environ.get("USE_4BIT", "").lower() in ("1", "true", "yes")
    lora_path = lora_path or os.environ.get("LORA_PATH", "").strip()
    if lora_path:
        lora_path = str(Path(lora_path).resolve())
        if not Path(lora_path).exists() or not (Path(lora_path) / "adapter_config.json").exists():
            lora_path = ""

    if use_4bit or lora_path:
        if lora_path:
            use_4bit = True
            if model_name == "google/flan-t5-small":
                model_name = "microsoft/Phi-3-mini-4k-instruct"
        model_kwargs = dict(
            quantization_config=_get_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        if "phi" in model_name.lower():
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
        tokenizer = AutoTokenizer.from_pretrained(
            lora_path if lora_path else model_name,
            trust_remote_code=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return pipe

    # Flan-T5 / seq2seq: newer transformers may not have "text2text-generation" in the registry
    try:
        return pipeline("text2text-generation", model=model_name)
    except KeyError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return _Seq2SeqWrapper(model, tokenizer)


def build_prompt(query: str, contexts) -> str:
    """Build RAG prompt from query and (score, text, metadata) context list."""
    ctx = "\n\n".join([f"[{i+1}] {c[1]}" for i, c in enumerate(contexts)])
    return (
        "You are a helpful policy assistant. Use only the provided context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n"
        "Answer concisely and cite sources like [1],[2]."
    )


class _Seq2SeqWrapper:
    """Wrapper for Seq2Seq (e.g. Flan-T5) when text2text-generation pipeline is not available."""

    task = "text2text-generation"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt, max_new_tokens=160, temperature=0.2, do_sample=True, clean_up_tokenization_spaces=True, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        if prompt and gen.startswith(prompt[:50]):
            gen = gen[len(prompt) :].strip()
        return [{"generated_text": gen}]


def generate_answer(
    pipe,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.2,
    do_sample: bool | None = None,
) -> str:
    """
    Run the pipeline and return the generated text.
    Works for both text2text (flan-t5) and text-generation (causal) pipelines.
    """
    do_sample = do_sample if do_sample is not None else (temperature > 0.0)
    task = getattr(pipe, "task", None)
    if task == "text2text-generation":
        out = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            clean_up_tokenization_spaces=True,
        )
        return out[0]["generated_text"]
    # Causal LM (e.g. Llama, Phi)
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    gen = out[0]["generated_text"]
    if prompt in gen:
        gen = gen[len(prompt) :].strip()
    return gen
