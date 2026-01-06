from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Loading T5 model... (this will download ~240MB on first run)")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
print("Model loaded successfully!")

# Test text
test_text = """
This Agreement is made between Party A and Party B on January 1, 2026.
Party A agrees to provide services to Party B for a period of 12 months.
The total payment shall be $50,000 paid in monthly installments.
Either party may terminate with 30 days written notice.
"""

print("\nOriginal text length:", len(test_text))
print("\nGenerating summary...")

# Prepare input
input_text = "summarize: " + test_text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\nSummary:", summary)
print("Summary length:", len(summary))
print("\n✅ Test successful! Ready to integrate.")