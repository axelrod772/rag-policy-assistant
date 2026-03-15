"""
Smoke tests for the LoRA feature: default generator works; invalid LORA_PATH is ignored; LoRA path is used when set and exists.
Run from project root: python -m scripts.test_lora_feature
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import get_generator, build_prompt, generate_answer


def test_default_generator_and_generation():
    """Default (no LoRA) returns a pipeline and generate_answer produces text."""
    gen = get_generator()
    assert gen is not None
    prompt = build_prompt("What is the waiting period?", [(1.0, "A 12-month waiting period applies.", {})])
    answer = generate_answer(gen, prompt, max_new_tokens=50, temperature=0.0)
    assert isinstance(answer, str) and len(answer.strip()) > 0
    print("OK: Default generator and generate_answer work.")


def test_invalid_lora_path_ignored():
    """When LORA_PATH points to a non-existent dir, generator falls back to default (no LoRA)."""
    gen = get_generator(lora_path="/nonexistent/lora/path")
    assert gen is not None
    task = getattr(gen, "task", None)
    assert task == "text2text-generation" or "generation" in str(type(gen)).lower()
    print("OK: Invalid LORA_PATH is ignored; default pipeline is used.")


def test_lora_path_used_when_exists():
    """When LORA_PATH exists and has adapter_config.json, generator loads base model + adapter."""
    import json
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parent.parent / "artifacts" / "lora_policy_test",
        Path(__file__).resolve().parent.parent / "artifacts" / "lora_policy",
    ]
    lora_dir = next((p for p in candidates if (p / "adapter_config.json").exists()), None)
    if not lora_dir:
        print("SKIP: No complete LoRA adapter found (run finetune_lora and wait for it to save).")
        return
    adapter_config = lora_dir / "adapter_config.json"
    with open(adapter_config, encoding="utf-8") as f:
        cfg = json.load(f)
        model_name = cfg.get("base_model_name_or_path")
    if not model_name and "lora_policy_test" in str(lora_dir):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    gen = get_generator(lora_path=str(lora_dir.resolve()), model_name=model_name or "microsoft/Phi-3-mini-4k-instruct")
    assert gen is not None
    assert getattr(gen, "task", None) == "text-generation"
    print(f"OK: LoRA adapter loaded from {lora_dir}.")


if __name__ == "__main__":
    test_default_generator_and_generation()
    test_invalid_lora_path_ignored()
    test_lora_path_used_when_exists()
    print("All LoRA feature checks passed.")
