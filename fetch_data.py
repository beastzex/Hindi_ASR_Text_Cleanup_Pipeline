"""
fetch_data.py

Fetches at least 25 transcription JSON and corresponding audio files from the
provided dataset. Saves them locally and creates a dataset index CSV.
"""

import os
import requests
import json
import logging
import pandas as pd
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_data(output_dir="data", num_samples=25):
    """
    Downloads metadata, audio, and reference transcripts.
    Saves them to the output directory and generates dataset_index.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_url = "https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI/export?format=csv&gid=1786138861"
    
    logging.info("Loading metadata from data/metadata.csv")
    try:
        df = pd.read_csv("data/metadata.csv")
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        return
    
    samples_collected = []
    success_count = 0
    
    for idx, row in df.iterrows():
        if success_count >= num_samples:
            break
            
        recording_id = row.get("recording_id")
        audio_url = str(row.get("rec_url_gcp")).replace("joshtalks-data-collection/hq_data/hi/", "upload_goai/")
        transcript_url = str(row.get("transcription_url_gcp")).replace("joshtalks-data-collection/hq_data/hi/", "upload_goai/")
        duration = row.get("duration", 0)
        
        if pd.isna(audio_url) or pd.isna(transcript_url):
            continue
            
        sample_id = f"sample_{success_count+1:03d}"
        
        # Determine file extensions
        audio_ext = os.path.splitext(urlparse(audio_url).path)[1]
        if not audio_ext:
            audio_ext = ".wav" # default if missing
            
        local_audio_path = os.path.join(output_dir, f"{sample_id}_audio{audio_ext}")
        local_ref_path = os.path.join(output_dir, f"{sample_id}_reference.txt")
        
        # Download Audio
        try:
            r_audio = requests.get(audio_url, timeout=15)
            r_audio.raise_for_status()
            with open(local_audio_path, 'wb') as f:
                f.write(r_audio.content)
        except Exception as e:
            logging.error(f"Try {idx+1}: Failed to download audio for {recording_id} - {e}")
            continue
            
        # Download Transcript JSON & Extract text
        try:
            r_trans = requests.get(transcript_url, timeout=15)
            r_trans.raise_for_status()
            trans_data = r_trans.json()
            
            # Extract transcript text. The JSON format is typically a list of dicts.
            if isinstance(trans_data, list):
                # concatenate all texts
                reference_text = " ".join([segment.get("text", "") for segment in trans_data if "text" in segment])
            elif isinstance(trans_data, dict) and "text" in trans_data:
                reference_text = trans_data["text"]
            elif isinstance(trans_data, dict) and "transcript" in trans_data:
                reference_text = trans_data["transcript"]
            else:
                reference_text = str(trans_data) # fallback
                
            # Clean up the text a bit
            reference_text = reference_text.replace('\n', ' ').strip()
            
            with open(local_ref_path, 'w', encoding='utf-8') as f:
                f.write(reference_text)
                
        except Exception as e:
            logging.error(f"Try {idx+1}: Failed to download or process transcript for {recording_id} - {e}")
            if os.path.exists(local_audio_path):
                os.remove(local_audio_path)
            continue
            
        # Log success and save index info
        samples_collected.append({
            "sample_id": sample_id,
            "audio_path": local_audio_path,
            "reference_transcript": local_ref_path,
            "duration": duration,
            "recording_id": recording_id
        })
        success_count += 1
        logging.info(f"[+] Successfully fetched {sample_id} (Recording {recording_id})")
        
    # Generate Index CSV
    index_path = os.path.join(output_dir, "dataset_index.csv")
    df_index = pd.DataFrame(samples_collected)
    df_index.to_csv(index_path, index=False)
    logging.info(f"Saved dataset index to {index_path} with {len(df_index)} samples.")

if __name__ == "__main__":
    fetch_data()
