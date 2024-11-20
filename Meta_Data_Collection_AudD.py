import json
import os
import pandas as pd
from collections import Counter
from datetime import datetime
import requests

class AudDMetaDataCollection:
    """
    Class for collecting metadata from audio files using AudD Api.

    Parameters:
        data (str): Data for api request. 
        files (list): List of audio file names.

    """
    def __init__(self, data, files):
        self.data = data
        self.files = files
        self.base_path = "./Data/MEMD_audio/"
        self.current_file_id = None

    def identify_song(self, file_path):
        """
        Identifies the song using AudD and returns the metadata as a dictionary.

        Parameters:
            file_path (str): The path to the audio file.

        Returns:
            dict or None: The metadata dictionary if successful, else None.
        """

        file = {
            'file': open(file_path, 'rb')
        }
        try:
            result = requests.post('https://api.audd.io/', data=self.data, files=file)
            result_dict = json.loads(result.text)
            return result_dict.get('result', {})
        except Exception as e:
            print(f"Error recognizing {file_path}: {e}")
            return None
        
    def extract_main_metadata(self, result_dict):
        """
        Extracts main metadata (label, artists, title, genres, release_date) from the result_dict.

        Parameters:
            result_dict (dict): The metadata dictionary returned by AudD.

        Returns:
            dict: A dictionary containing extracted main metadata.
        """
        metadata = {
            'audio_file': self.current_file_id,
            "label": "Not Found",
            "artists": "Not Found",
            "title": "Not Found",
            "genres": "Not Found",
            "release_date": "Not Found"
        }

        metadata["artist"] = result_dict.get("artist", "Not Found")
        metadata["title"] = result_dict.get("title", "Not Found")
        metadata["genres"] = result_dict.get("genreNames", "Not Found")
        metadata["release_date"] = result_dict.get("release_date", "Not Found")
        metadata["label"] = result_dict.get("label", "Not Found")
        
        return metadata
        
    def process_audio_files(self, output_csv):
        """
        Processes all audio files in the specified directory and writes the results to a CSV.

        Parameters:
            audio_directory (str): The path to the directory containing audio files.
            output_csv (str): The path where the CSV file will be saved.
        """

        metadata = []

        for file in self.files['audio_file']:
            print(f"Processing: {file}")

            self.current_file_id = file
            
            file_path = self.base_path + file
            result_dict = self.identify_song(file_path)

            if not result_dict:
                continue
            # Extract main metadata
            main_metadata = self.extract_main_metadata(result_dict)
            metadata.append(main_metadata)
            
        df = pd.DataFrame(metadata)
        if os.path.exists(output_csv):
            print("Exist")
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(output_csv, index=False)

        print(f"Metadata saved to {output_csv}")
