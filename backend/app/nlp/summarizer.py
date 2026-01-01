from typing import Optional
import re

def summarize_text_smart(text: str, length: str = "medium") -> str:
    """
    Intelligent paragraph-based summarization that extracts complete 
    paragraphs and key sections from legal documents.
    
    Args:
        text: The document text to summarize
        length: "short", "medium", or "long"
    
    Returns:
        A formatted summary with complete paragraphs
    """
    if not text or len(text.strip()) == 0:
        return "⚠️ No text available to summarize."
    
    # Clean the text
    text = text.strip()
    
    # Split into paragraphs (handle both \n\n and single \n)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    # If no double newlines, split by single newlines
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 20]
    
    # Define limits based on length preference
    limits = {
        "short": 1000,      # ~150 words
        "medium": 2500,     # ~350 words
        "long": 5000        # ~700 words
    }
    
    char_limit = limits.get(length, 2500)
    
    # Prioritize important paragraphs (those with legal keywords)
    important_keywords = [
        'agreement', 'party', 'parties', 'hereby', 'shall', 'terms',
        'conditions', 'obligations', 'rights', 'pursuant', 'effective',
        'termination', 'liability', 'indemnification', 'confidential',
        'whereas', 'therefore', 'notwithstanding'
    ]
    
    # Score paragraphs by importance
    scored_paragraphs = []
    for para in paragraphs:
        score = sum(1 for keyword in important_keywords if keyword.lower() in para.lower())
        scored_paragraphs.append((score, para))
    
    # Sort by score (descending) but keep original order for high-scored paragraphs
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Extract paragraphs until we hit the limit
    summary_parts = []
    total_chars = 0
    
    for score, para in scored_paragraphs:
        para_length = len(para)
        
        # If adding this paragraph would exceed limit significantly, skip it
        if total_chars + para_length > char_limit and total_chars > 0:
            continue
        
        # If this is the first paragraph and it's too long, truncate it
        if total_chars == 0 and para_length > char_limit:
            truncated = para[:char_limit].rsplit('.', 1)[0] + "."
            summary_parts.append(truncated)
            total_chars += len(truncated)
            break
        
        summary_parts.append(para)
        total_chars += para_length
        
        # If we've got enough content, stop
        if total_chars >= char_limit * 0.8:  # 80% of limit is good enough
            break
    
    if not summary_parts:
        # Fallback: just take first paragraph
        first_para = paragraphs[0] if paragraphs else text
        if len(first_para) > char_limit:
            first_para = first_para[:char_limit].rsplit('.', 1)[0] + "."
        return first_para
    
    # Join paragraphs with proper spacing
    summary = "\n\n".join(summary_parts)
    
    # Add metadata footer
    word_count = len(summary.split())
    para_count = len(summary_parts)
    
    summary += f"\n\n---\n📄 Summary: {para_count} key section(s) | {word_count} words"
    
    # Add indicator if there's more content
    if len(summary) < len(text) * 0.9:  # If summary is less than 90% of original
        summary += " | Full document contains additional details"
    
    return summary


def summarize_text_mock(text: str, length: str = "short") -> str:
    """
    Legacy simple summarizer - kept for backward compatibility
    """
    if not text:
        return "No text available to summarize."
    
    limits = {
        "short": 500,
        "medium": 2000,
        "long": 5000
    }
    
    char_limit = limits.get(length, 2000)
    
    if len(text) <= char_limit:
        return text
    
    return text[:char_limit] + "..."


# For future: Real AI-powered summarization
"""
from transformers import pipeline

class AISummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        if len(text) < 100:
            return text
        
        # Split into chunks if text is too long
        max_chunk_size = 1024
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        summaries = []
        for chunk in chunks[:5]:  # Limit to 5 chunks
            result = self.summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        
        return " ".join(summaries)
"""