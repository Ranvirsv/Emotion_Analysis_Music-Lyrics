import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans

class MuLanEmotionAnalyzer:
    def __init__(self, embeddings_file):
        with open(embeddings_file, 'r') as f:
            self.data = json.load(f)
            self.song_ids = list(self.data.keys())
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.emotion_map = {
            'happy': ['happy', 'excited', 'joyful'],
            'sad': ['sad', 'depressed', 'gloomy'],
            'angry': ['angry', 'frustrated', 'irritated'],
            'calm': ['calm', 'relaxed', 'peaceful']
        }
        
    def get_embeddings_arrays(self):
        audio_embeddings = []
        lyrics_embeddings = []

        for song_id, embeddings in self.data.items():
            audio_embeddings.append(np.array(embeddings['audio_embedding']))
            lyrics_embeddings.append(np.array(embeddings['text_embedding']))

        audio_embeddings = np.array(audio_embeddings)
        lyrics_embeddings = np.array(lyrics_embeddings)
        
        audio_embeddings = np.squeeze(audio_embeddings)
        lyrics_embeddings = np.squeeze(lyrics_embeddings)
            
        return audio_embeddings, lyrics_embeddings
    
    def analyze_emotion_alignment(self):
        audio_emb, lyrics_emb = self.get_embeddings_arrays()
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(audio_emb, lyrics_emb)
        
        # Get diagonal (matching pairs) similarities
        matching_similarities = np.diag(similarity_matrix)
        
        # Get emotion labels for lyrics
        lyrics_emotions = self.get_lyrics_emotions(self.song_ids)
        
        # Assign emotion labels to audio embeddings
        audio_emotions = self.cluster_audio_embeddings(audio_emb)
        
        if audio_emotions is not None:
            # Compute emotion alignment scores
            emotion_alignment_scores = []
            for i in range(len(matching_similarities)):
                audio_emotion = audio_emotions[i]
                lyrics_emotion = lyrics_emotions[i]
                emotion_alignment_score = self.emotion_alignment_score(audio_emotion, lyrics_emotion)
                emotion_alignment_scores.append(emotion_alignment_score)
        else:
            emotion_alignment_scores = [0.0] * len(matching_similarities)
        
        results = {
            'similarity_matrix': similarity_matrix,
            'emotion_alignment_scores': emotion_alignment_scores,
            'best_emotion_match': {
                'song_id': self.song_ids[np.argmax(emotion_alignment_scores)],
                'score': np.max(emotion_alignment_scores)
            },
            'worst_emotion_match': {
                'song_id': self.song_ids[np.argmin(emotion_alignment_scores)],
                'score': np.min(emotion_alignment_scores)
            },
            'lyrics_emotions': lyrics_emotions,
            'audio_emotions': audio_emotions
        }
        
        return results
    
    def get_lyrics_emotions(self, song_ids):
        """
        Extract the dominant emotion for each song's lyrics.
        
        Parameters:
        song_ids (list): A list of song IDs.
        
        Returns:
        list: A list of emotion labels ('happy', 'sad', 'angry', 'calm') for each song.
        """
        lyrics_emotions = []
        for song_id in song_ids:
            embeddings = self.data[song_id]
            lyrics = embeddings['text_embedding']
            lyrics_str = ' '.join([str(x) for x in lyrics])
            sentiment_scores = self.sentiment_analyzer.polarity_scores(lyrics_str)
            emotion = self.get_dominant_emotion(sentiment_scores)
            lyrics_emotions.append(emotion)
        return lyrics_emotions
    
    def get_dominant_emotion(self, sentiment_scores):
        """
        Map the sentiment scores to dominant emotion categories.
        
        Parameters:
        sentiment_scores (dict): A dictionary containing the sentiment scores (positive, negative, neutral, compound).
        
        Returns:
        str: The dominant emotion category ('happy', 'sad', 'angry', or 'calm').
        """
        # Map the compound sentiment score to specific emotion categories
        if sentiment_scores['compound'] >= 0.5:
            return 'happy'
        elif sentiment_scores['compound'] <= -0.5:
            return 'sad'
        elif sentiment_scores['neg'] > sentiment_scores['pos']:
            return 'angry'
        else:
            return 'calm'
    
    def cluster_audio_embeddings(self, audio_embeddings):
        """
        Cluster the audio embeddings and extract the dominant emotion for each cluster.
        
        Parameters:
        audio_embeddings (np.ndarray): The audio embeddings to be clustered.
        
        Returns:
        list: A list of emotion labels ('happy', 'sad', 'angry', 'calm') for each audio embedding.
        """
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        audio_clusters = kmeans.fit_predict(audio_embeddings)
        
        audio_emotions = []
        for cluster_label in audio_clusters:
            if cluster_label == 0:
                audio_emotions.append('happy')
            elif cluster_label == 1:
                audio_emotions.append('sad')
            elif cluster_label == 2:
                audio_emotions.append('angry')
            else:
                audio_emotions.append('calm')
        
        return audio_emotions

    def emotion_alignment_score(self, audio_emotion, lyrics_emotion):
        """
        Calculate the alignment score between the audio and lyrics emotions.
        
        Parameters:
        audio_emotion (str): The emotion label for the audio.
        lyrics_emotion (str): The emotion label for the lyrics.
        
        Returns:
        float: The alignment score (1.0 if the emotions match, 0.0 otherwise).
        """
        if audio_emotion == lyrics_emotion:
            return 1.0
        else:
            return 0.0
    
    def plot_emotion_alignment(self, results):
        """
        Create visualizations to display the emotion alignment results.
        
        Parameters:
        results (dict): The results from the `analyze_emotion_alignment` method.
        """
        plt.figure(figsize=(15, 8))

        # 1. Emotion Distribution Heatmap
        plt.subplot(2, 2, 1)
        emotion_alignment_matrix = np.zeros((len(self.emotion_map), len(self.emotion_map)))
        for i, lyrics_emotion in enumerate(self.emotion_map):
            for j, audio_emotion in enumerate(self.emotion_map):
                scores = [results['emotion_alignment_scores'][k] for k, (l, a) in enumerate(zip(results['lyrics_emotions'], results['audio_emotions'])) if l == lyrics_emotion and a == audio_emotion]
                if scores:
                    emotion_alignment_matrix[i, j] = np.mean(scores)
        
        sns.heatmap(emotion_alignment_matrix, 
                   cmap='YlOrRd',
                   xticklabels=list(self.emotion_map.keys()),
                   yticklabels=list(self.emotion_map.keys()),
                   annot=True, fmt='.2f')
        plt.title('Emotion Alignment Matrix')
        plt.xlabel('Audio Emotion')
        plt.ylabel('Lyrics Emotion')


        # 2. Emotion Alignment Scores Histogram
        plt.subplot(2, 2, 2)
        plt.hist(results['emotion_alignment_scores'], bins=20)
        plt.xlabel('Emotion Alignment Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Emotion Alignment Scores')

        # 3. Best and Worst Aligned Songs
        plt.subplot(2, 2, 3)
        x = np.arange(len(self.song_ids))
        plt.bar(x, results['emotion_alignment_scores'])
        plt.axhline(y=results['best_emotion_match']['score'], 
                   color='g', 
                   linestyle='--', 
                   label=f"Best: {results['best_emotion_match']['song_id']}")
        plt.axhline(y=results['worst_emotion_match']['score'],
                   color='r',
                   linestyle='--',
                   label=f"Worst: {results['worst_emotion_match']['song_id']}")
        plt.xlabel('Song Index')
        plt.ylabel('Emotion Alignment Score')
        plt.title('Per-Song Emotion Alignment Scores')
        plt.legend()

        # 4. Correlation between Similarity and Alignment
        plt.subplot(2, 2, 4)
        similarity_scores = np.diag(results['similarity_matrix'])
        plt.scatter(similarity_scores, results['emotion_alignment_scores'])
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Emotion Alignment Score')
        plt.title('Correlation between Similarity and Alignment')

        plt.tight_layout()
        plt.show()
