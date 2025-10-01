from transformers import pipeline

def get_generator(model_name: str = "google/flan-t5-small"):
    return pipeline("text2text-generation", model=model_name)

def build_prompt(query: str, contexts):
    ctx = "\n\n".join([f"[{i+1}] {c[1]}" for i, c in enumerate(contexts)])
    prompt = (
        "You are a helpful assistant. Use only the provided context to answer.\n"
        "If the answer is not in the context, say you don't know.\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\n"
        "Answer concisely and cite sources like [1],[2]."
    )
    return prompt
