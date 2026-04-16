# Hindi ASR Text Cleanup Pipeline

This project implements an end-to-end Hindi Automatic Speech Recognition (ASR) text cleanup pipeline. It is designed to process raw conversational Hindi audio transcripts, clean and normalize the output by converting numbers into their corresponding Hindi words, and detecting English words embedded in the Hindi sentences (code-switching).

## Features

- **Data Fetching:** Automatically downloads and prepares a sample dataset of Hindi audio files.
- **ASR Generation:** Generates raw speech-to-text transcriptions for the sample audios.
- **Hindi Number Normalization:** Converts numeric digits (e.g., `10`, `100`, `1997`) scattered in the transcript to their proper text forms in Hindi (e.g., `दस`, `सौ`, `उन्नीस सौ सत्तानवे`).
- **English Word Tagging:** Identifies and appropriately tags English words occurring in conversational Hindi (Hinglish/code-switched language scenarios).
- **Report Generation:** Automatically generates a comprehensive Markdown report summarizing the results.

## Requirements

The core dependencies for this project are listed in `requirements.txt`:
- `torch`, `transformers`, `datasets`
- `librosa`, `soundfile`
- `requests`, `pandas`, `indic-nlp-library`
- `langdetect`, `tqdm`, `jiwer`, `numpy`

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Task_2
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the complete pipeline from data fetching to report generation, simply execute the `main.py` file:

```bash
python main.py
```

### Execution Flow

When you run `main.py`, it executes the following steps sequentially:
1. **Fetching data:** Downloads sample audio data into the `data/` directory.
2. **Generating raw ASR:** Extracts and transcribes audio to raw Hindi text.
3. **Running cleanup pipeline:** Processes texts via the `ASRCleanupPipeline` to normalize numbers and tag English words.
4. **Generating report:** Validates outputs and compiles a final `REPORT.md` inside the `output/` directory containing the results and examples of the cleaned text.

## Modules Overview

- `main.py`: Main entry point that orchestrates the entire pipeline.
- `fetch_data.py`: Handles downloading and preparing audio dataset.
- `generate_asr.py`: Extracts and generates baseline ASR transcriptions.
- `pipeline.py`: The wrapper class responsible for orchestrating the text cleanup steps.
- `hindi_normalizer.py`: Component designed strictly to normalize numbers to Hindi textual equivalents.
- `english_tagger.py`: Component for detecting and logging or tagging English words within Hindi sentences.
- `report_generator.py`: Automates result summarization.

## Note on Version Control
The `data/` and `output/` directories are ignored in version control to prevent storing large audio datasets and generated output reports in the repository. Executing `main.py` creates these directories automatically.
