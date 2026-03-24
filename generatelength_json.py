import wave
import json
import glob
import os

def generate_wav_json():
    # The exact folder you specified
    target_dir = "/storage/ssd1/richtsai1103/MusicBench/egs/musicbench"
    
    # Verify the folder actually exists before trying to scan it
    if not os.path.exists(target_dir):
        print(f"Error: The directory {target_dir} does not exist or you lack permission.")
        return

    # Look specifically for .wav files in that folder
    search_pattern = os.path.join(target_dir, "*.wav")
    wav_files = glob.glob(search_pattern)
    
    if not wav_files:
        print(f"No .wav files found in {target_dir}.")
        return

    print(f"Found {len(wav_files)} .wav files. Processing...")
    audio_data = {}

    for file_path in wav_files:
        # Extract just the filename (e.g., 'song.wav') instead of the giant path
        file_name = os.path.basename(file_path)
        
        try:
            with wave.open(file_path, 'r') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration_seconds = frames / float(rate)
                
                audio_data[file_name] = round(duration_seconds, 2)
        except Exception as e:
            audio_data[file_name] = f"Error: {e}"

    # Save the output JSON file directly into that same SSD folder
    output_file = os.path.join(target_dir, "wav_durations.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(audio_data, json_file, indent=4)
        print(f"Success! Data organized and saved to: {output_file}")
    except OSError as e:
        print(f"\nError saving file (your SSD might also be full): {e}")
        # Fallback to printing directly to your terminal
        print("\nHere is the raw data instead:\n")
        print(json.dumps(audio_data, indent=4))

if __name__ == "__main__":
    generate_wav_json()