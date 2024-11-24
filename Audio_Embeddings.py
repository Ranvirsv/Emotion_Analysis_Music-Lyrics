import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2Model
import os
import json
from tqdm import tqdm

class MultimodalEmbeddingGenerator:
    def __init__(self):
        print("Initializing models...")
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')

        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder.to(self.device)
        self.audio_encoder.to(self.device)

        self.text_encoder.eval()
        self.audio_encoder.eval()
    
    def process_audio(self, audio_path):
        """Process audio file for wav2vec2"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process segments
            segment_length = 16000 * 30  # 30 seconds
            segments = torch.split(waveform, segment_length, dim=1)
            
            # Remove last segment if too short
            segments = [s for s in segments if s.shape[1] > 16000]  # At least 1 second
            
            return segments
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {str(e)}")
            return None

    def get_audio_embedding(self, audio_path):
        """Extract audio embeddings using Wav2Vec2"""
        try:
            with torch.no_grad():
                segments = self.process_audio(audio_path)
                if not segments:
                    return None
                
                segment_embeddings = []
                for segment in segments:
                    # Process audio with wav2vec2
                    inputs = self.audio_processor(segment.squeeze(), 
                                                sampling_rate=16000, 
                                                return_tensors="pt",
                                                padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.audio_encoder(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    segment_embeddings.append(embedding)
                
                # Average embeddings across segments
                final_embedding = torch.mean(torch.stack(segment_embeddings), dim=0)
                return final_embedding.cpu().numpy()
                
        except Exception as e:
            print(f"Error getting audio embedding for {audio_path}: {str(e)}")
            return None

    def get_text_embedding(self, lyrics_path):
        """Extract text embeddings using BERT"""
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read()
            
            with torch.no_grad():
                inputs = self.text_tokenizer(lyrics, 
                                           return_tensors="pt", 
                                           padding=True, 
                                           truncation=True, 
                                           max_length=512)
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.text_encoder(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                
                return embedding.cpu().numpy().squeeze()
                
        except Exception as e:
            print(f"Error getting text embedding for {lyrics_path}: {str(e)}")
            return None

    def create_multimodal_embedding(self, audio_path, lyrics_path):
        """Create combined embeddings for audio and lyrics"""
        audio_embedding = self.get_audio_embedding(audio_path)
        text_embedding = self.get_text_embedding(lyrics_path)
        
        if audio_embedding is None or text_embedding is None:
            return None
        
        # Calculate similarity
        similarity = np.dot(audio_embedding.flatten(), text_embedding.flatten()) / \
                    (np.linalg.norm(audio_embedding) * np.linalg.norm(text_embedding))
        
        return {
            'audio_embedding': audio_embedding.tolist(),
            'text_embedding': text_embedding.tolist(),
            'similarity': float(similarity)
        }

    def process_batch(self, audio_dir, lyrics_dir, output_file):
        """Process a batch of audio-lyrics pairs"""
        embeddings = {}
        
        # Get matching audio and lyrics files
        audio_files = {os.path.splitext(f)[0]: os.path.join(audio_dir, f) 
                      for f in os.listdir(audio_dir) if f.endswith(('.mp3', '.wav'))}
        lyrics_files = {os.path.splitext(f)[0]: os.path.join(lyrics_dir, f) 
                       for f in os.listdir(lyrics_dir) if f.endswith('.txt')}
        
        common_ids = set(audio_files.keys()) & set(lyrics_files.keys())
        print(f"Found {len(common_ids)} matching audio-lyrics pairs")
        
        for file_id in tqdm(common_ids, desc="Processing files"):
            try:
                embedding = self.create_multimodal_embedding(
                    audio_files[file_id],
                    lyrics_files[file_id]
                )
                if embedding:
                    embeddings[file_id] = embedding
            except Exception as e:
                print(f"Error processing {file_id}: {str(e)}")
        
        # Save embeddings
        with open(output_file, 'w') as f:
            json.dump(embeddings, f)
        
        return embeddings

def main():
    # Initialize the generator
    generator = MultimodalEmbeddingGenerator()
    
    # Define directories
    audio_dir = "/Users/user/adv_nlp_project/Emotion_Analysis_Music-Lyrics/audio_files"
    lyrics_dir = "/Users/user/adv_nlp_project/Emotion_Analysis_Music-Lyrics/lyrics"
    output_file = "audio_lyrics_embeddings.json"
    
    # Process files
    embeddings = generator.process_batch(audio_dir, lyrics_dir, output_file)
    
    # Print statistics
    print(f"\nProcessing complete:")
    print(f"Processed {len(embeddings)} files successfully")
    if embeddings:
        sample_id = next(iter(embeddings))
        print(f"Audio embedding dimension: {len(embeddings[sample_id]['audio_embedding'])}")
        print(f"Text embedding dimension: {len(embeddings[sample_id]['text_embedding'])}")
        print(f"Average similarity score: {np.mean([e['similarity'] for e in embeddings.values()]):.4f}")

if __name__ == "__main__":
    main()