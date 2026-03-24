import os
import json
import soundfile as sf
from datasets import load_dataset, Audio # Add Audio here
from tqdm import tqdm

# --- CONFIGURATION ---
os.environ['HF_HOME'] = "/storage/ssd1/richtsai1103/hf_cache"
output_dir = "/storage/ssd1/richtsai1103/Jamendo_Test"
audio_dir = os.path.join(output_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

num_samples = 5 

print(f"Streaming {num_samples} samples...")
# Load the dataset
dataset = load_dataset("amaai-lab/JamendoMaxCaps", split="train", streaming=True)

# FORCE the decoding to use a standard format (bypasses torchcodec)
dataset = dataset.cast_column("audio", Audio(sampling_rate=44100)) 

subset = dataset.take(num_samples)

metadata_records = []

for i, example in enumerate(tqdm(subset, total=num_samples)):
    track_id = example.get('id', str(i))
    start_t = int(example.get('start_time', 0))
    file_name = f"{track_id}_{start_t}.wav"
    file_path = os.path.join(audio_dir, file_name)
    
    # Save Audio
    sf.write(file_path, example['audio']['array'], example['audio']['sampling_rate'])
    
    # Metadata Logic (Same as your script)
    raw_tags = example.get('tag', [])
    genres = [t.split('---')[1] for t in raw_tags if 'genre' in t]
    instruments = [t.split('---')[1] for t in raw_tags if 'instrument' in t]
    moods = [t.split('---')[1] for t in raw_tags if 'mood' in t]

    record = {
        "key": "", 
        "artist": example.get("artist_name", "Jamendo Artist"),
        "sample_rate": example['audio']['sampling_rate'],
        "file_extension": "wav",
        "description": example.get("caption", ""),
        "keywords": ", ".join(genres + instruments + moods),
        "duration": 30.0,
        "bpm": example.get("musicinfo", {}).get("tempo", ""),
        "genre": genres[0] if genres else "instrumental", 
        "title": example.get("track_name", f"Track {track_id}"),
        "name": f"{track_id}_{start_t}",
        "instrument": ", ".join(instruments) if instruments else "Mix",
        "moods": moods,
        "start_time": float(start_t),
        "end_time": float(example.get('end_time', start_t + 30))
    }
    metadata_records.append(record)

with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
    for record in metadata_records:
        f.write(json.dumps(record) + "\n")

print(f"\nDone! Test files saved to {output_dir}")