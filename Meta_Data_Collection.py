import json
import os
import pandas as pd
from collections import Counter
from datetime import datetime

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

    def extract_main_metadata(self, result_dict):
        """
        Extracts main metadata (label, artists, title, genres, release_date) from the result_dict.

        Parameters:
            result_dict (dict): The metadata dictionary returned by ACRCloud.

        Returns:
            dict: A dictionary containing extracted main metadata.
        """
        main_metadata = {
            "label": "Not Found",
            "artists": "Not Found",
            "title": "Not Found",
            "genres": "Not Found",
            "release_date": "Not Found"
        }

        try:
            music = result_dict.get("metadata", {}).get("music", [])
            if not music:
                return main_metadata

            first_music = music[0]

            # Extract label
            label = first_music.get("label")
            if label:
                main_metadata["label"] = label.strip()

            # Extract artists
            artists = first_music.get("artists", [])
            artist_names = [artist.get("name", "").strip() for artist in artists if artist.get("name")]
            if artist_names:
                main_metadata["artists"] = "; ".join(artist_names)

            # Extract title
            title = first_music.get("title")
            if title:
                main_metadata["title"] = title.strip()

            # Extract genres
            genres = first_music.get("genres", [])
            genre_names = [genre.get("name", "").strip() for genre in genres if genre.get("name")]
            if genre_names:
                main_metadata["genres"] = "; ".join(genre_names)

            # Extract and format release_date
            release_date_str = first_music.get("release_date")
            if release_date_str:
                try:
                    # Attempt to parse the release_date
                    release_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                    main_metadata["release_date"] = release_date.strftime("%Y-%m-%d")
                except ValueError:
                    # If parsing fails, keep the original string
                    main_metadata["release_date"] = release_date_str.strip()

        except Exception as e:
            print(f"Error extracting main metadata: {e}")

        return main_metadata

    def extract_backup_metadata(self, result_dict):
        """
        Extracts (artist, title) pairs from external metadata sources as backup.

        Parameters:
            result_dict (dict): The metadata dictionary returned by ACRCloud.

        Returns:
            list of tuples: A list containing (artist, title) tuples from external sources.
        """
        metadata = []
        try:
            music = result_dict.get("metadata", {}).get("music", [])
            if not music:
                return metadata

            external_metadata = music[0].get("external_metadata", {})
            sources = ["deezer", "spotify", "youtube"]

            for source in sources:
                try:
                    source_data = external_metadata.get(source)
                    if not source_data:
                        continue

                    # For sources like Deezer and Spotify
                    if source in ["deezer", "spotify"]:
                        track = source_data.get("track", {})
                        artists_info = track.get("artists", [])
                        artist_name = artists_info[0].get("name", "").strip() if artists_info else None
                        title_name = track.get("name", "").strip()

                        if artist_name and title_name:
                            metadata.append((artist_name, title_name))

                    # For YouTube, assume 'track' key may not be present
                    elif source == "youtube":
                        # YouTube metadata might differ; adapt accordingly
                        # Example: Assuming 'title' contains both artist and song title separated by a dash
                        vid = source_data.get("vid", "")
                        # Without specific structure, we cannot extract artist/title from YouTube
                        # So, we skip or implement custom logic if available
                        continue

                except (KeyError, IndexError, TypeError):
                    continue

        except Exception as e:
            print(f"Error extracting backup metadata: {e}")

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
        for root, _, files in os.walk(audio_directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    result_dict = self.identify_song(file_path)

                    if (result_dict and 
                        "metadata" in result_dict and 
                        "music" in result_dict["metadata"] and 
                        len(result_dict["metadata"]["music"]) > 0):

                        # Extract main metadata
                        main_metadata = self.extract_main_metadata(result_dict)

                        # Initialize artist and title from main metadata
                        artist = main_metadata.get("artists", "Not Found")
                        title = main_metadata.get("title", "Not Found")

                        # If artist or title is not found in main metadata, use backup
                        if artist == "Not Found" or title == "Not Found":
                            backup_metadata = self.extract_backup_metadata(result_dict)
                            backup_artist, backup_title = self.get_most_common_metadata(backup_metadata)

                            # Replace only if main metadata is missing
                            if artist == "Not Found":
                                artist = backup_artist
                            if title == "Not Found":
                                title = backup_title

                        # Extract genres and release_date from main metadata
                        genres = main_metadata.get("genres", "Not Found")
                        release_date = main_metadata.get("release_date", "Not Found")
                        label = main_metadata.get("label", "Not Found")

                    else:
                        artist, title = ("Not Found", "Not Found")
                        genres = "Not Found"
                        release_date = "Not Found"
                        label = "Not Found"

                    results.append({
                        "audio_file": file,
                        "artist": artist,
                        "title": title,
                        "genres": genres,
                        "release_date": release_date,
                        "label": label
                    })

        # Create a DataFrame and write to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Metadata written to {output_csv}")
