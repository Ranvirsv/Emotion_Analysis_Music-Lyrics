import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words
STOP_WORDS = set(stopwords.words('english'))

def preprocess_lyrics(file_path):
    """
    Clean and tokenize the lyrics from a file.
    
    Args:
        file_path (str): Path to the lyrics file.
    
    Returns:
        list: A list of cleaned and tokenized words from the lyrics.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Normalize text: lowercase
    text = text.lower()
    
    # Remove unwanted markers and special characters
    text = re.sub(r'\[.*?\]', '', text)  # Removes [Intro], [Verse], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Removes (spoken words) or other brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keeps only letters and spaces
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return filtered_tokens

def process_lyrics_folder(input_folder, output_folder):
    """
    Process all lyrics files in a folder and save cleaned versions to a new folder.
    
    Args:
        input_folder (str): Path to the folder containing raw lyrics files.
        output_folder (str): Path to the folder where processed lyrics will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            # Process the lyrics
            processed_tokens = preprocess_lyrics(file_path)
            # Save the processed lyrics to a new file
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(' '.join(processed_tokens))
