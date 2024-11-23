import os
import torch
import numpy as np
from musiclm_pytorch import TextTransformer, AudioSpectrogramTransformer
from transformers import AutoTokenizer
import torchaudio

# Initialize the text transformer (MuLaN Text component)
text_transformer = TextTransformer(
    dim=512, depth=6, heads=8, dim_head=64
)

# Initialize the audio transformer (MuLaN Audio component)
audio_transformer = AudioSpectrogramTransformer(
    dim=512, depth=6, heads=8, dim_head=64, spec_n_fft=128, spec_win_length=24
)

# Initialize a tokenizer for text processing
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use a relevant tokenizer

def process_and_save_lyrics_embeddings(input_folder, output_file):
    """
    Process all lyrics in a folder, extract embeddings, and save them to a file.
    Args:
        input_folder (str): Path to folder containing cleaned lyric files.
        output_file (str): Path to save embeddings.
    """
    embeddings = []
    song_ids = []

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            # Read the cleaned lyrics
            with open(file_path, 'r', encoding='utf-8') as file:
                lyrics = file.read()

            # Tokenize and split into chunks of max length 256
            tokens = tokenizer(
                lyrics,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False  # No padding for chunks
            )

            input_ids_chunks = tokens["input_ids"].split(dim=1, split_size=256)  # Split into 256-token chunks

            # Generate embedding for each chunk and average them
            chunk_embeddings = []
            with torch.no_grad():
                for chunk in input_ids_chunks:
                    chunk_embedding = text_transformer(chunk)
                    chunk_embeddings.append(chunk_embedding)

            # Average embeddings across chunks
            aggregated_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
            embeddings.append(aggregated_embedding.detach().numpy())
            song_ids.append(file_name.replace('.txt', ''))
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save embeddings and song IDs
    np.save(output_file, {'song_ids': song_ids, 'embeddings': embeddings})
    print(f"Lyrics embeddings saved to {output_file}")


def process_and_save_audio_embeddings(input_folder, output_file):
    """
    Process all audio files in a folder, extract embeddings, and save them to a file.
    Args:
        input_folder (str): Path to folder containing audio files.
        output_file (str): Path to save embeddings.
    """
    embeddings = []
    song_ids = []

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.wav'):
            # Load audio file
            wav, _ = torchaudio.load(file_path)

            # Generate embedding
            with torch.no_grad():
                embedding = audio_transformer(wav)
                embeddings.append(embedding.detach().numpy())
                song_ids.append(file_name.replace('.wav', ''))

    # Save embeddings and song IDs
    np.save(output_file, {'song_ids': song_ids, 'embeddings': embeddings})
    print(f"Audio embeddings saved to {output_file}")

def process_all_embeddings(lyrics_folder, audio_folder, lyrics_output, audio_output):
    """
    Process both lyrics and audio, saving their embeddings.
    Args:
        lyrics_folder (str): Path to the folder with cleaned lyrics.
        audio_folder (str): Path to the folder with audio files.
        lyrics_output (str): Path to save lyric embeddings.
        audio_output (str): Path to save audio embeddings.
    """
    print("Processing lyrics...")
    process_and_save_lyrics_embeddings(lyrics_folder, lyrics_output)
    print("Processing audio...")
    process_and_save_audio_embeddings(audio_folder, audio_output)
    print("Processing complete!")
