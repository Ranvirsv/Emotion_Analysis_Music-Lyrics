import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import openl3
import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa
from sentence_transformers import SentenceTransformer


def extract_audio_embeddings(
    csv_path,
    audio_folder,
    output_csv,
    content_type="music",
    embedding_size=512,
    hop_size=0.1
):
    """
    Extracts audio embeddings using OpenL3 from processed audio segments listed in the CSV.

    Parameters:
    - csv_path (str): Path to the CSV file containing audio file information.
    - audio_folder (str): Path to the folder containing processed audio segments.
    - output_csv (str): Path to save the extracted embeddings.
    - content_type (str): "music" or "environment".
    - embedding_size (int): 512 or 6144, depending on the model.
    - hop_size (float): Hop size in seconds for frame-level embeddings.

    Returns:
    - None. Saves the embeddings to the specified CSV file.
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} entries.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Ensure 'audio_file' column exists
    if 'audio_file' not in df.columns:
        print("Error: 'audio_file' column not found in CSV.")
        return

    audio_files = df['audio_file'].tolist()

    embeddings_data = []

    # Iterate through each audio file
    for audio_file in tqdm(audio_files, desc="Processing Audio Files"):
        base_name = os.path.splitext(audio_file)[0]
        # Find all segments for this base audio file
        segment_files = [
            f for f in os.listdir(audio_folder)
            if f.startswith(base_name) and f.endswith(".wav")
        ]

        if not segment_files:
            print(f"No segments found for {audio_file}. Skipping.")
            continue

        for segment_file in segment_files:
            segment_path = os.path.join(audio_folder, segment_file)
            try:
                # Load audio
                audio, sr = sf.read(segment_path)

                # Ensure audio is mono and at the correct sample rate
                if sr != 48000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                    sr = 48000
                    print(f"Resampled {segment_file} to {sr} Hz.")

                # Extract frame-level embeddings
                embeddings, _ = openl3.get_audio_embedding(
                    audio,
                    sr,
                    content_type=content_type,
                    embedding_size=embedding_size,
                    hop_size=hop_size,
                    center=True
                )

                if embeddings.size == 0:
                    print(f"No embeddings extracted for {segment_file}. Skipping.")
                    continue

                # Aggregate embeddings (mean pooling)
                embedding_mean = embeddings.mean(axis=0)

                # Convert to space-separated string for CSV storage
                embedding_str = ' '.join(map(str, embedding_mean))

                # Append to data list
                embeddings_data.append({
                    "audio_segment": segment_file,
                    "embedding": embedding_str
                })

            except Exception as e:
                print(f"Error processing {segment_file}: {e}")

    if not embeddings_data:
        print("No embeddings were extracted. Please check your audio files and preprocessing steps.")
        return

    # Convert to DataFrame
    embeddings_df = pd.DataFrame(embeddings_data)

    # Save to CSV
    try:
        embeddings_df.to_csv(output_csv, index=False)
        print(f"Embeddings successfully saved to {output_csv}.")
    except Exception as e:
        print(f"Error saving embeddings to CSV: {e}")
        
def extract_text_embeddings(
    csv_path,
    lyrics_folder,
    output_csv,
    model_name='all-MiniLM-L6-v2',
    batch_size=32
):
    """
    Extracts text embeddings using SentenceTransformer from lyrics stored in separate files.

    Parameters:
    - csv_path (str): Path to the CSV file containing audio file metadata.
    - lyrics_folder (str): Path to the folder containing cleaned lyrics files.
    - output_csv (str): Path to save the extracted lyrics embeddings.
    - model_name (str): Name of the pre-trained SentenceTransformer model.
    - batch_size (int): Number of samples to process in a batch.

    Returns:
    - None. Saves the embeddings to the specified CSV file.
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} entries.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Ensure 'audio_file' column exists
    if 'audio_file' not in df.columns:
        print("Error: 'audio_file' column not found in CSV.")
        return

    audio_files = df['audio_file'].tolist()

    # Initialize the SentenceTransformer model
    try:
        model = SentenceTransformer(model_name)
        print(f"Loaded SentenceTransformer model: {model_name}")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return

    # Initialize lists to store data
    lyrics_data = []
    embedding_data = []

    # Iterate through each audio file to process corresponding lyrics
    for audio_file in tqdm(audio_files, desc="Processing Lyrics"):
        base_name = os.path.splitext(audio_file)[0]
        # Define expected lyrics file name (e.g., "1157.txt")
        lyrics_file = f"{base_name}.txt"
        lyrics_path = os.path.join(lyrics_folder, lyrics_file)

        if not os.path.exists(lyrics_path):
            print(f"Lyrics file {lyrics_file} not found in {lyrics_folder}. Skipping.")
            continue

        try:
            # Read lyrics
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                cleaned_lyrics = f.read()

            # Append to lists
            lyrics_data.append(cleaned_lyrics)

        except Exception as e:
            print(f"Error reading {lyrics_file}: {e}")
            continue

    if not lyrics_data:
        print("No lyrics were processed. Please check your lyrics files and paths.")
        return

    # Generate text embeddings in batches
    try:
        embeddings = model.encode(lyrics_data, batch_size=batch_size, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating text embeddings: {e}")
        return

    # Prepare DataFrame
    embeddings_df = pd.DataFrame({
        "audio_file": [os.path.splitext(f)[0] for f in audio_files if os.path.exists(os.path.join(lyrics_folder, f"{os.path.splitext(f)[0]}.txt"))],
        "lyrics_embedding": [' '.join(map(str, emb)) for emb in embeddings]
    })

    # Save to CSV
    try:
        embeddings_df.to_csv(output_csv, index=False)
        print(f"Lyrics embeddings successfully saved to {output_csv}.")
    except Exception as e:
        print(f"Error saving lyrics embeddings to CSV: {e}")



def load_audio_embeddings(embeddings_csv):
    """
    Loads embeddings from a CSV file and converts them into a Pandas DataFrame with NumPy arrays.

    Parameters:
    - embeddings_csv (str): Path to the embeddings CSV file.

    Returns:
    - pd.DataFrame: DataFrame with 'audio_segment' and 'embedding' columns.
    """
    try:
        df = pd.read_csv(embeddings_csv)
        # Convert the embedding strings back to NumPy arrays
        df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x, sep=' '))
        print(f"Loaded {len(df)} embeddings from {embeddings_csv}.")
        return df
    except Exception as e:
        print(f"Error loading embeddings CSV: {e}")
        return pd.DataFrame()

def load_lyrics_embeddings(embeddings_csv):
    """
    Loads lyrics embeddings from a CSV file and converts them into a Pandas DataFrame with NumPy arrays.

    Parameters:
    - embeddings_csv (str): Path to the lyrics embeddings CSV file.

    Returns:
    - pd.DataFrame: DataFrame with 'audio_file' and 'lyrics_embedding' columns.
    """
    try:
        df = pd.read_csv(embeddings_csv)
        df['lyrics_embedding'] = df['lyrics_embedding'].apply(lambda x: np.fromstring(x, sep=' '))
        print(f"Loaded {len(df)} lyrics embeddings from {embeddings_csv}.")
        return df
    except Exception as e:
        print(f"Error loading lyrics embeddings CSV: {e}")
        return pd.DataFrame()
