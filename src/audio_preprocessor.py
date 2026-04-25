"""
Audio preprocessing module
Handles: loading, resampling, noise reduction, segmentation
"""

import librosa
import numpy as np
import noisereduce as nr
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, target_sr=16000, min_duration=10, max_duration=120):
        """
        Initialize audio preprocessor
        
        Args:
            target_sr: Target sample rate in Hz
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
        """
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def load_audio(self, file_path):
        """
        Load audio file with resampling
        
        Args:
            file_path: Path to audio file
            
        Returns:
            audio: numpy array of audio samples
            sr: sample rate
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def remove_noise(self, audio, sr):
        """
        Remove background noise using spectral subtraction
        
        Args:
            audio: audio signal
            sr: sample rate
            
        Returns:
            denoised_audio: cleaned audio signal
        """
        # Estimate noise from first 0.5 seconds (assuming it's silence/background)
        noise_sample = audio[:int(0.5 * sr)]
        if len(noise_sample) > 0:
            denoised_audio = nr.reduce_noise(
                y=audio, 
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=0.8
            )
        else:
            denoised_audio = audio
        
        return denoised_audio
    
    def trim_silence(self, audio, sr, top_db=25):
        """
        Trim leading/trailing silence
        
        Args:
            audio: audio signal
            sr: sample rate
            top_db: threshold in decibels
            
        Returns:
            trimmed_audio: audio with silence removed
        """
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio
    
    def normalize_audio(self, audio):
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: audio signal
            
        Returns:
            normalized_audio: normalized signal
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalized_audio = audio / max_val
        else:
            normalized_audio = audio
        return normalized_audio
    
    def segment_audio(self, audio, sr, segment_duration=10, overlap=0.5):
        """
        Segment long audio into overlapping chunks
        
        Args:
            audio: audio signal
            sr: sample rate
            segment_duration: duration of each segment in seconds
            overlap: overlap ratio between segments
            
        Returns:
            segments: list of audio segments
        """
        segment_samples = int(segment_duration * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
        
        return segments
    
    def preprocess(self, file_path, segment=True):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path: path to audio file
            segment: whether to segment long audio
            
        Returns:
            processed_audio: preprocessed audio signal(s)
            metadata: dictionary with processing info
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None, None
        
        original_duration = len(audio) / sr
        
        # Check duration
        if original_duration < self.min_duration:
            print(f"Warning: {file_path} is too short ({original_duration:.1f}s)")
            return None, None
        
        # Trim silence
        audio = self.trim_silence(audio, sr)
        
        # Remove noise
        audio = self.remove_noise(audio, sr)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        metadata = {
            'original_duration': original_duration,
            'processed_duration': len(audio) / sr,
            'sample_rate': sr
        }
        
        # Segment if requested
        if segment and len(audio) / sr > self.max_duration:
            segments = self.segment_audio(audio, sr)
            return segments, metadata
        else:
            return audio, metadata
    
    def extract_basic_info(self, file_path):
        """
        Extract basic audio information without full preprocessing
        
        Args:
            file_path: path to audio file
            
        Returns:
            info: dictionary with audio metadata
        """
        try:
            duration = librosa.get_duration(filename=file_path)
            return {
                'duration': duration,
                'file_path': str(file_path),
                'file_name': file_path.name if hasattr(file_path, 'name') else file_path
            }
        except:
            return None

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    print("AudioPreprocessor initialized successfully!")
    print(f"  - Target sample rate: {preprocessor.target_sr} Hz")
    print(f"  - Min duration: {preprocessor.min_duration}s")
    print(f"  - Max duration: {preprocessor.max_duration}s")