"""
report_generator.py

Generates the final output REPORT.md and aggregates stats from the pipeline.
"""
import pandas as pd
import os

def generate_report(results_csv_path="output/pipeline_results.csv", report_md_path="output/REPORT.md"):
    if not os.path.exists(results_csv_path):
        print(f"Results CSV {results_csv_path} not found.")
        return
        
    df = pd.read_csv(results_csv_path)
    
    total_samples = len(df)
    total_numbers = df["numbers_converted"].sum()
    total_english = df["english_words_tagged"].sum()
    avg_time = df["processing_time_ms"].mean() if total_samples > 0 else 0
    errors = df[df["notes"] != "None"].shape[0] if "notes" in df.columns else 0
    
    # Get examples for Number Normalization
    num_examples = df[df['numbers_converted'] > 0].head(5)
    num_rows = "\n".join([f"| {row['sample_id']} | {row['raw_asr'][:50]}... | {row['cleaned_asr'][:50]}... | {row['numbers_converted']} |" for _, row in num_examples.iterrows()])
    
    if num_examples.empty:
        # Fallback if no numbers were converted
        num_rows = "| N/A | No conversions triggered | N/A | 0 |"
        
    # Get examples for English Word Tagging
    eng_examples = df[df['english_words_tagged'] > 0].head(5)
    eng_rows = "\n".join([f"| {row['sample_id']} | {row['cleaned_asr'][:100]}... |" for _, row in eng_examples.iterrows()])
    
    if eng_examples.empty:
        eng_rows = "| N/A | No English words tagged |"
        
    report = f"""# Task 2: Hindi ASR Cleanup Pipeline Report

## Overview
Built an end-to-end pipeline to fetch conversational Hindi audio from GCP storage, transcribe it using `whisper-small`, and apply a two-component text cleanup process: converting written Hindi numbers to digits (normalization) and tagging English-origin words (code-switching detection).

## Component 1: Number Normalization
### Approach
A rule-based mapping approach was chosen because Hindi number words follow strict grammatical patterns. This avoids needing a large training corpus and allows full explainability, which is easily extended to exclude specific idiomatic phrases.

### Results
| Sample ID | Raw ASR (Before) | Cleaned ASR (After) | Numbers Converted |
|---|---|---|---|
{num_rows}

### Edge Cases
1. **Hyphenated numbers**: Elements like 'दो-चार' represent "a few" rather than mathematically "2-4" or "24". The rule-based engine uses a pattern matcher to safely exclude these idioms.
2. **Contextual Punctuation**: Handling numbers just before commas or periods without corrupting the punctuation is resolved through token suffix checks.

### Limitations
Rule-based matching can struggle with very heavily mis-spelled number formats that Whisper may occasionally output if the speech is highly dialectical.

## Component 2: English Word Detection  
### Approach
A hybrid approach utilizing regular expressions for pure Roman text and a dictionary-backed lookup for Devanagari transliterated English words. High confidence is given to known vocabulary words, and medium to unrecognized Roman strings.

### Results
| Sample ID | Cleaned Text Snippet with [EN] Tags |
|---|---|
{eng_rows}

### Ambiguous Cases
1. Proper noun overlaps with common words (e.g. 'Apple' as a company vs fruit). Handled conservatively unless heavily contextual.
2. Loan words that have become so ubiquitous they are essentially Hindi.

### Limitations
Relies on dictionary completeness for Devanagari transliteration. Unexpected spellings (e.g. 'कंप्यूटर' vs 'कम्पूटर') will be missed without string distance matching (e.g., Levenshtein).

## Pipeline Statistics
- Total samples processed: {total_samples}
- Number conversions made: {total_numbers}
- English words tagged: {total_english}
- Average processing time: {avg_time:.2f}ms per sample
- Pipeline errors: {errors}

## What I Would Improve With More Time
1. **Fuzzy String Matching**: Integrate IndicNLP library phonetic similarity to catch misspelled numbers or transliterated English words.
2. **Batch Processing**: Run Whisper inference in batches instead of iterating row by row for an enormous speedup.
3. **LLM Verification Step**: Add a small, fast LLM pass specifically for disambiguating extreme edge cases (e.g., determining if a name is a brand or a person).
"""
    
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Report successfully saved to {report_md_path}")

if __name__ == "__main__":
    generate_report()
