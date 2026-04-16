"""
main.py

The main entry point for the Hindi ASR text cleanup pipeline.
Executes all steps sequentially.
"""
import os
import logging
from fetch_data import fetch_data
from generate_asr import generate_asr
from pipeline import ASRCleanupPipeline
from report_generator import generate_report

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("\n--- Step 1: Fetching data... ---")
    fetch_data(output_dir="data", num_samples=25)
    
    print("\n--- Step 2: Generating raw ASR... ---")
    generate_asr(index_csv_path="data/dataset_index.csv")
    
    print("\n--- Step 3: Running cleanup pipeline... ---")
    pipeline = ASRCleanupPipeline()
    pipeline.process_dataset("data/dataset_index.csv", "output/pipeline_results.csv")
    
    print("\n--- Step 4: Generating report... ---")
    generate_report(results_csv_path="output/pipeline_results.csv", report_md_path="output/REPORT.md")
    
    print("\nDone! Check Task_2/output/ folder for results.")

if __name__ == "__main__":
    main()
