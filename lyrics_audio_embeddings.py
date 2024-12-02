import os
import torch
import numpy as np
from musiclm_pytorch import TextTransformer, AudioSpectrogramTransformer
from transformers import AutoTokenizer
import torchaudio

# Initialize the text transformer (MuLaN Text component)
text_transformer = TextTransformer(dim=512, depth=6, heads=8, dim_head=64)

# Initialize the audio transformer (MuLaN Audio component)
audio_transformer = AudioSpectrogramTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    spec_n_fft=1024,
    spec_win_length=1024,
    spec_hop_length=512
)

# Initialize a tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def setup_audio_backend():
    """Set up the appropriate audio backend"""
    try:
        torchaudio.set_audio_backend("sox_io")
    except Exception as e:
        print(f"Audio backend setup failed: {e}")
    return "cuda" if torch.cuda.is_available() else "cpu"

def process_audio(file_path, device):
    """Process a single audio file and return its embedding"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Normalize waveform
        waveform = waveform / (waveform.abs().max() + 1e-8)
        waveform = waveform.to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = audio_transformer(waveform)
            return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_and_save_audio_embeddings(input_folder, output_file):
    """Process all audio files and save embeddings"""
    device = setup_audio_backend()
    audio_transformer.to(device)
    audio_transformer.eval()

    embeddings = []
    song_ids = []
    mp3_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]

    for idx, file_name in enumerate(mp3_files, 1):
        file_path = os.path.join(input_folder, file_name)
        embedding = process_audio(file_path, device)
        if embedding is not None:
            embeddings.append(embedding)
            song_ids.append(file_name.replace('.mp3', ''))
            print(f"Successfully processed {file_name}")
        else:
            print(f"Failed to process {file_name}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, song_ids=song_ids, embeddings=embeddings)
    print(f"Audio embeddings saved to {output_file}")

def process_and_save_lyrics_embeddings(input_folder, output_file):
    """Process all lyrics files and save embeddings"""
    embeddings = []
    song_ids = []

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                lyrics = file.read()

            tokens = tokenizer(
                lyrics,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False
            )
            input_ids_chunks = torch.split(tokens["input_ids"], 256, dim=1)

            chunk_embeddings = []
            with torch.no_grad():
                for chunk in input_ids_chunks:
                    chunk_embedding = text_transformer(chunk)
                    chunk_embeddings.append(chunk_embedding)

            aggregated_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
            embeddings.append(aggregated_embedding.numpy())
            song_ids.append(file_name.replace('.txt', ''))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez(output_file, song_ids=song_ids, embeddings=embeddings)
    print(f"Lyrics embeddings saved to {output_file}")

def process_all_embeddings(lyrics_folder, audio_folder, lyrics_output, audio_output):
    """Process both lyrics and audio embeddings"""
    print("Processing lyrics...")
    process_and_save_lyrics_embeddings(lyrics_folder, lyrics_output)
    print("Processing audio...")
    process_and_save_audio_embeddings(audio_folder, audio_output)
    print("Processing complete!")
