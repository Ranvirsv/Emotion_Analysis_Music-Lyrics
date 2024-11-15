import json
import os
import pandas as pd
from collections import Counter

class MetaDataCollection:
    def __init__(self, recognizer):
        """
        Initializes the MetaDataCollection with a given ACRCloudRecognizer instance.

        Parameters:
            recognizer (ACRCloudRecognizer): An instance of ACRCloudRecognizer.
        """
        self.recognizer = recognizer

    def identify_song(self, file_path):
        """
        Identifies the song using ACRCloud and returns the metadata as a dictionary.

        Parameters:
            file_path (str): The path to the audio file.

        Returns:
            dict or None: The metadata dictionary if successful, else None.
        """
        try:
            result = self.recognizer.recognize_by_file(file_path, 0)
            result_dict = json.loads(result)
            return result_dict
        except Exception as e:
            print(f"Error recognizing {file_path}: {e}")
            return None

    def extract_metadata(self, result_dict):
        """
        Extracts (artist, title) pairs from all available metadata sources.

        Parameters:
            result_dict (dict): The metadata dictionary returned by ACRCloud.

        Returns:
            list of tuples: A list containing (artist, title) tuples.
        """
        metadata = []
        try:
            music = result_dict.get("metadata", {}).get("music", [])
            if not music:
                return metadata

            external_metadata = music[0].get("external_metadata", {})
            sources = ["deezer", "spotify", "youtube", "isrc", "upc"]

            for source in sources:
                source_data = external_metadata.get(source)
                if source_data:
                    # Adjust the extraction based on the source's data structure
                    # Assuming 'track' key exists for all sources except ISRC and UPC
                    if source in ["deezer", "spotify", "youtube"]:
                        track = source_data.get("track", {})
                        artist_info = track.get("artists", [])
                        if artist_info:
                            artist = artist_info[0].get("name")
                            title = track.get("name")
                            if artist and title:
                                metadata.append((artist.strip(), title.strip()))
                    elif source == "isrc":
                        # ISRC might not have artist/title directly; handle accordingly
                        isrc_info = source_data.get("isrc")
                        if isrc_info:
                            artist = isrc_info.get("artist")
                            title = isrc_info.get("title")
                            if artist and title:
                                metadata.append((artist.strip(), title.strip()))
                    elif source == "upc":
                        # UPC might require additional processing or may not contain artist/title
                        upc_info = source_data.get("upc")
                        if upc_info:
                            artist = upc_info.get("artist")
                            title = upc_info.get("title")
                            if artist and title:
                                metadata.append((artist.strip(), title.strip()))
        except Exception as e:
            print(f"Error extracting metadata: {e}")

        return metadata

    def get_most_common_metadata(self, metadata_list):
        """
        Determines the most common (artist, title) pair from the metadata list.

        Parameters:
            metadata_list (list of tuples): A list of (artist, title) tuples.

        Returns:
            tuple: The most common (artist, title) pair, or ("Not Found", "Not Found") if empty.
        """
        if not metadata_list:
            return ("Not Found", "Not Found")
        
        counter = Counter(metadata_list)
        most_common = counter.most_common(1)
        if most_common:
            return most_common[0][0]
        else:
            return ("Not Found", "Not Found")

    def process_audio_files(self, audio_directory, output_csv):
        """
        Processes all audio files in the specified directory and writes the results to a CSV.

        Parameters:
            audio_directory (str): The path to the directory containing audio files.
            output_csv (str): The path where the CSV file will be saved.
        """
        # Supported audio file extensions
        supported_extensions = ('.mp3', '.wav', '.flac', '.aac', '.m4a')

        # List to store results
        results = []

        # Iterate over all files in the directory
        for root, dirs, files in os.walk(audio_directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    result_dict = self.identify_song(file_path)
                    
                    if (result_dict and 
                        "metadata" in result_dict and 
                        "music" in result_dict["metadata"] and 
                        len(result_dict["metadata"]["music"]) > 0):
                        
                        metadata_list = self.extract_metadata(result_dict)
                        artist, title = self.get_most_common_metadata(metadata_list)
                    else:
                        artist, title = ("Not Found", "Not Found")
                    
                    results.append({
                        "audio_file": file,
                        "artist": artist,
                        "title": title
                    })

        # Create a DataFrame and write to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Metadata written to {output_csv}")
