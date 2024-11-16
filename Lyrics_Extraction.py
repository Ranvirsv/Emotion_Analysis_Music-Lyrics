import requests
from bs4 import BeautifulSoup
import time

class LyricsExtractor:
    def __init__(self, genius_api_token):
        self.base_url = "https://api.genius.com"
        self.headers = {"Authorization": f"Bearer {genius_api_token}"}
    
    def search_song(self, title, artist):
        """
        Search for a song on Genius using its title and artist.
        Returns the URL to the song's lyrics page if found, else None.
        """
        search_url = f"{self.base_url}/search"
        params = {"q": f"{title} {artist}"}
        
        response = requests.get(search_url, headers=self.headers, params=params)
        if response.status_code != 200:
            print(f"Error: Genius API returned status code {response.status_code}")
            return None
        
        # Look for matches where the artist matches
        hits = response.json().get("response", {}).get("hits", [])
        for hit in hits:
            if artist.lower() in hit["result"]["primary_artist"]["name"].lower():
                return hit["result"]["url"]
        return None

    def scrape_lyrics(self, url):
        """
        Scrape lyrics from a Genius lyrics page URL.
        Returns the lyrics as a string or None if not found.
        """
        if not url:
            return None
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Could not fetch the lyrics page (status {response.status_code}).")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        # Try to locate the lyrics container
        lyrics_containers = soup.find_all("div", {"data-lyrics-container": "true"})
        if lyrics_containers:
            return "\n".join([container.get_text(separator="\n").strip() for container in lyrics_containers])
        else:
            print(f"Error: Could not find lyrics container on page: {url}")
        return None

    def fetch_lyrics(self, title, artist):
        """
        High-level function to fetch lyrics by searching and scraping.
        Combines search_song() and scrape_lyrics().
        """
        try:
            song_url = self.search_song(title, artist)
            if song_url:
                lyrics = self.scrape_lyrics(song_url)
                return lyrics
            else:
                print(f"Lyrics not found for '{title}' by '{artist}'.")
                return None
        except Exception as e:
            print(f"Error fetching lyrics for '{title}' by '{artist}': {e}")
            return None
