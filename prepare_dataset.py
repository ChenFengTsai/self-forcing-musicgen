import json
import os
import wave
import random
from pathlib import Path
from tqdm import tqdm

INPUT_JSON = '/storage/ssd1/richtsai1103/MusicBench/MusicBench_train.json'
AUDIO_BASE_DIR = '/storage/ssd1/richtsai1103/MusicBench/datashare/data'
BASE_OUTPUT_DIR = '/storage/ssd1/richtsai1103/MusicBench/egs/musicbench'

def process_split(split_name, dataset_chunk):
    print(f"\nProcessing {split_name} split ({len(dataset_chunk)} items)...")
    split_dir = os.path.join(BASE_OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    jsonl_path = os.path.join(split_dir, 'data.jsonl')
    jsonl_entries = []
    
    for item in tqdm(dataset_chunk):
        json_location = item['location']
        audio_path = os.path.join(AUDIO_BASE_DIR, json_location)
        
        if not os.path.exists(audio_path):
            filename_only = os.path.basename(json_location)
            audio_path = os.path.join(AUDIO_BASE_DIR, filename_only)
            
        if not os.path.exists(audio_path):
            continue
            
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
        except Exception:
            continue
        
        captions = {"main": item.get("main_caption", ""), "alt": item.get("alt_caption", "")}
        
        for cap_type, caption_text in captions.items():
            if not caption_text.strip():
                continue
                
            base_name = f"{Path(audio_path).stem}_{cap_type}"
            symlink_path = os.path.join(split_dir, f"{base_name}.wav")
            json_path = os.path.join(split_dir, f"{base_name}.json")
            
            if not os.path.exists(symlink_path):
                os.symlink(audio_path, symlink_path)
            
            metadata = {
                "key": item.get("prompt_key", "").replace("The key of this song is ", "").strip("."),
                "artist": "", "sample_rate": sample_rate, "file_extension": "wav",
                "description": caption_text, "keywords": "", "duration": duration,
                "bpm": item.get("prompt_bpm", "").replace("The bpm is ", "").strip("."),
                "genre": "", "title": "", "name": Path(audio_path).stem,
                "instrument": "", "moods": []
            }
            
            with open(json_path, 'w') as mf:
                json.dump(metadata, mf)
                
            jsonl_entries.append({
                "path": symlink_path, 
                "duration": duration,
                "sample_rate": sample_rate,
                "amplitude": None,
                "weight": None
            })

    with open(jsonl_path, 'w') as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + '\n')
            
    return len(jsonl_entries)

def main():
    print(f"Loading dataset from: {INPUT_JSON}")
    try:
        with open(INPUT_JSON, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError:
        with open(INPUT_JSON, 'r') as f:
            dataset = [json.loads(line) for line in f]

    # SHUFFLE AND SPLIT THE DATASET
    random.seed(42) # For reproducibility
    random.shuffle(dataset)
    
    total = len(dataset)
    train_end = int(total * 0.90)
    valid_end = int(total * 0.95)
    
    train_data = dataset[:train_end]
    valid_data = dataset[train_end:valid_end]
    generate_data = dataset[valid_end:]
    
    train_count = process_split('train', train_data)
    valid_count = process_split('valid', valid_data)
    gen_count = process_split('generate', generate_data)

    print(f"\nDone! Generated entries - Train: {train_count}, Valid: {valid_count}, Generate: {gen_count}")

if __name__ == "__main__":
    main()