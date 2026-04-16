"""
pipeline.py

Integrates the ASR texts with the two cleanup components:
1. HindiNumberNormalizer
2. EnglishWordTagger

Outputs metrics and complete processed results.
"""
import time
import pandas as pd
import logging
from hindi_normalizer import HindiNumberNormalizer
from english_tagger import EnglishWordTagger
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ASRCleanupPipeline:
    def __init__(self):
        self.normalizer = HindiNumberNormalizer()
        self.tagger = EnglishWordTagger()
        
    def process(self, raw_asr: str, reference: str) -> dict:
        start_time = time.time()
        
        # 1. Number Normalization
        try:
            norm_text, norm_changes = self.normalizer.normalize(raw_asr)
        except Exception as e:
            logging.error(f"Error in Number Normalizer: {e}")
            norm_text = raw_asr
            norm_changes = []
            
        # 2. English Word Tagging
        try:
            tagged_text, tag_detections = self.tagger.tag(norm_text)
        except Exception as e:
            logging.error(f"Error in English Word Tagger: {e}")
            tagged_text = norm_text
            tag_detections = []
            
        end_time = time.time()
        proc_time_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "cleaned_asr": tagged_text,
            "intermediate_num_norm": norm_text,
            "numbers_converted": len(norm_changes),
            "english_words_tagged": len(tag_detections),
            "processing_time_ms": proc_time_ms,
            "norm_changes": norm_changes,
            "tag_detections": tag_detections,
            "error": "None"
        }
        
    def process_dataset(self, index_csv_path: str, output_csv_path: str) -> pd.DataFrame:
        if not os.path.exists(index_csv_path):
            logging.error(f"Index CSV {index_csv_path} not found.")
            return pd.DataFrame()
            
        df = pd.read_csv(index_csv_path)
        
        results = []
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running Pipeline"):
            sample_id = row.get("sample_id", "Unknown")
            raw_asr_path = row.get("raw_asr_path", "")
            ref_path = row.get("reference_transcript", "")
            
            # Read files safely
            try:
                with open(raw_asr_path, "r", encoding="utf-8") as f:
                    raw_asr = f.read()
                with open(ref_path, "r", encoding="utf-8") as f:
                    reference = f.read()
            except Exception as e:
                logging.error(f"Sample {sample_id}: File missing or read error. {e}")
                continue
                
            res = self.process(raw_asr, reference)
            
            results.append({
                "sample_id": sample_id,
                "reference": reference,
                "raw_asr": raw_asr,
                "cleaned_asr": res["cleaned_asr"],
                "numbers_converted": res["numbers_converted"],
                "english_words_tagged": res["english_words_tagged"],
                "processing_time_ms": res["processing_time_ms"],
                "notes": res["error"]
            })
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_csv_path, index=False)
        logging.info(f"Pipeline processing complete. Saved to {output_csv_path}")
        return df_res

if __name__ == "__main__":
    pipeline = ASRCleanupPipeline()
    pipeline.process_dataset("data/dataset_index.csv", "output/pipeline_results.csv")
