from typing import Optional
import re
from transformers import BartTokenizer, BartForConditionalGeneration
import gc
import torch 

#Global model cache (loaded once on a startup)

_model = None
_tokenizer = None

def get_model():
    """Lazy load the Bart model (only loads once)"""
    global _model, _tokenizer
    if _model is None:
        print("Loadking BART model...")
        _tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        _model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
        print("BART model loaded successfully!!")
    return _model, _tokenizer


def extract_legal_sections(text: str):
    """
    Segragating before summarizing : 
    """
    sections = {
        "issues": [],
        "facts": [],
        "reasoning": [],
        "directions": []
    }

    lines = text.split("\n")

    for line in lines:
        l = line.lower()

        if any(k in l for k in ["issue", "petition", "writ", "challenge", "claims"]):
            sections["issues"].append(line)

        elif any(k in l for k in ["background", "facts", "leased", "declared", "forest", "bbtcl"]):
            sections["facts"].append(line)

        elif any(k in l for k in ["we hold", "we consider", "therefore", "in view of", "court"]):
            sections["reasoning"].append(line)

        elif any(k in l for k in [
            "we direct", "cec", "shall", "is directed", "12 weeks",
            "listed on", "survey", "report", "restoration"
        ]):
            sections["directions"].append(line)

    return sections


def chunk_by_tokens(text, tokenizer, max_tokens=900):
    """Chunks text by sentences to avoid enconding entire doc at once"""
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        #Encode just the sentence
        sent_tokens = tokenizer.encode(sentence, truncation=True, max_length=max_tokens)
        sent_length = len(sent_tokens)

        #If adding this sentence exceed limit, save current chunk
        if current_length + sent_length > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            #Keep last sentence for context overlapping
            current_chunk = [current_chunk[-1]] if current_chunk else []
            current_length = len(tokenizer.encode(current_chunk[0])) if current_chunk else 0

        current_chunk.append(sentence)
        current_length += sent_length

    #Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_directions_verbatim(text):
    directions = []
    lines = text.split("\n")

    capture = False
    para_pattern = re.compile(r'^\d+\.\s')
    for line in lines:
        l = line.lower().strip()

        # Start capturing after court begins issuing directions
        if any(k in l for k in ["we hereby direct", "we direct", "in view of", "it is directed"]):
            capture = True

        if capture:
            if para_pattern.match(line.strip()) and any(k in l for k in [
                "cec", "shall", "directed", "granted", "listed", "survey", "list on", "listed on"
            ]):
                if len(l) > 20:
                    directions.append(line.strip())
                    continue
            if any(k in l for k in [
                "cec", "shall", "weeks", "listed on",
                "survey", "report", "rehabilitation",
                "restoration", "time is granted","appeal allowed",
                "appeal dismissed","judgment reversed",
                "set aside","deduction disallowed","assessee not entitled",
                "high court erred"

            ]):
                if len(l) > 20:   # avoid headings
                    directions.append(line.strip())

    # Remove duplicates
    return list(dict.fromkeys(directions))



def summarize_text_ai(text: str, length: str = "medium") -> str:
    if not text or len(text.strip()) == 0:
        return "No text available to summarize."

    try:
        model, tokenizer = get_model()

        # 1. Extract binding court directions (verbatim)
        directions = extract_directions_verbatim(text)
        #chunks = chunk_by_tokens(clean_text, tokenizer, max_tokens=900)


        # 2. Remove directions from AI input
        clean_text = text
        for d in directions:
            clean_text = clean_text.replace(d, "")

        # 3. Summarize only the non-binding background
        chunks = chunk_by_tokens(clean_text, tokenizer, max_tokens=900)

        summaries = []
        for chunk in chunks[:5]:   # safety limit
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=220,
                min_length=100,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True
            )

            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

        combined = " ".join(summaries)
        
        # 4. Final refinement pass
        if len(combined.split()) > 400:
            final_input = tokenizer(
                "Summarize this legal judgement concisely, preserving key facts and reasoning:\n" + combined,
                return_tensors = "pt",
                truncation=True,
                max_length=1024
            )

            final_ids = model.generate(
                final_input["input_ids"],
                max_length = 250,
                min_length = 150,
                num_beams = 4,
                length_penalty = 1.2
            )
            final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)

        else:
            final_summary = combined
        # 5. Attach binding court order verbatim
        return f"""
            CASE SUMMARY

            {final_summary}

            ---

            COURT DIRECTIONS (Binding Order)

            """ + "\n".join("• " + d for d in directions) + """

            ---
            🤖 AI-Legal-Documenter
            """

    except Exception as e:
        print(f"AI summarization failed : {e}")
        return summarize_text_mock(text, length)



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