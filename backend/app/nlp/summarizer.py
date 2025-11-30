from typing import Optional

# Lightweight mock summarizer for offline dev
def summarize_text_mock(text: str, length: str = "short") -> str:
    if not text:
        return "No text available to summarize."
    # very naive: return the first N chars/lines
    if length == "short":
        return text[:300] + ("..." if len(text) > 300 else "")
    if length == "medium":
        return text[:700] + ("..." if len(text) > 700 else "")
    return text[:1500] + ("..." if len(text) > 1500 else "")

#TO DO: replace with HF transformers pipeline when ready:
# from transformers import pipeline
# summarizer = pipeline("summarization", model="t5-small")
# def summarize_text_hf(text, length): ...