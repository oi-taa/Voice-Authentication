"""
Adaptive Threshold Selection for Voice Authentication
"""
import numpy as np
import os
import pickle
from scipy.spatial.distance import pdist, cosine


class AdaptiveThresholdSystem:
    """
    Implements adaptive threshold selection based on enrollment quality.
    """
    
    def __init__(self, base_threshold=0.45, min_threshold=0.30, max_threshold=0.60):
        """
        Initialize adaptive threshold system.
        
        Args:
            base_threshold: Default threshold for medium-quality enrollments
            min_threshold: Minimum threshold (most permissive)
            max_threshold: Maximum threshold (most strict)
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.user_thresholds = {}  # Store per-user thresholds
        
    def calculate_enrollment_consistency(self, feature_vectors):
        """
        Calculate how consistent the enrollment samples are.
        
        Args:
            feature_vectors: numpy array of feature vectors from enrollment
            
        Returns:
            consistency_score: float between 0 (inconsistent) and 1 (very consistent)
        """
        if len(feature_vectors) < 2:
            return 0.5  # Neutral for single sample
        
        # Calculate pairwise distances
        try:
            pairwise_distances = pdist(feature_vectors, metric='cosine')
        except:
            # Fallback if pdist fails
            distances = []
            n = len(feature_vectors)
            for i in range(n):
                for j in range(i+1, n):
                    dist = cosine(feature_vectors[i], feature_vectors[j])
                    distances.append(dist)
            pairwise_distances = np.array(distances)
        
        mean_dist = np.mean(pairwise_distances)
        std_dist = np.std(pairwise_distances)
        
        # Convert distance to consistency score
        # Lower mean distance = higher consistency
        consistency = 1.0 - min(mean_dist * 2, 1.0)
        
        # Penalize high variance
        variance_penalty = min(std_dist * 3, 0.2)
        consistency = max(0, consistency - variance_penalty)
        
        return consistency
    
    def get_adaptive_threshold(self, username, enrollment_features=None, save=True):
        """
        Get or calculate adaptive threshold for a user.
        
        Args:
            username: user identifier
            enrollment_features: feature vectors from enrollment (if available)
            save: whether to save the calculated threshold
            
        Returns:
            threshold: personalized threshold for this user
            quality: enrollment quality score
        """
        # Check if we have a stored threshold for this user
        if username in self.user_thresholds and enrollment_features is None:
            return self.user_thresholds[username]['threshold'], \
                   self.user_thresholds[username]['quality']
        
        # Calculate new threshold based on enrollment quality
        if enrollment_features is not None:
            consistency = self.calculate_enrollment_consistency(enrollment_features)
            
            # Adjust threshold based on consistency
            # High consistency -> stricter threshold (higher value)
            # Low consistency -> looser threshold (lower value)
            adjustment = (consistency - 0.5) * 0.3  # ±0.15 adjustment
            adaptive_threshold = self.base_threshold + adjustment
            
            # Clamp to allowed range
            adaptive_threshold = np.clip(adaptive_threshold, 
                                        self.min_threshold, 
                                        self.max_threshold)
            
            if save:
                self.user_thresholds[username] = {
                    'threshold': adaptive_threshold,
                    'quality': consistency,
                    'samples': len(enrollment_features)
                }
            
            return adaptive_threshold, consistency
        
        # Fallback to base threshold
        return self.base_threshold, 0.5
    
    def calculate_confidence(self, similarity_score, threshold):
        """
        Calculate confidence in the authentication decision.
        
        Args:
            similarity_score: similarity between test and enrollment
            threshold: decision threshold
            
        Returns:
            confidence: float between 0 and 1
        """
        # How far from the threshold?
        margin = abs(similarity_score - threshold)
        
        # Normalize by reasonable margin
        confidence = min(margin / 0.2, 1.0)
        
        return confidence
    
    def get_quality_feedback(self, quality_score):
        """
        Generate human-readable feedback about enrollment quality.
        
        Args:
            quality_score: enrollment consistency score
            
        Returns:
            feedback: string with quality assessment
        """
        if quality_score >= 0.75:
            return "EXCELLENT - Very consistent enrollment. High security level."
        elif quality_score >= 0.55:
            return "GOOD - Acceptable enrollment quality."
        elif quality_score >= 0.35:
            return "FAIR - Consider re-recording in a quieter environment."
        else:
            return "POOR - Please re-record. Find a quiet location and speak clearly."
    
    def save_thresholds(self, filepath='user_thresholds.pkl'):
        """Save user thresholds to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.user_thresholds, f)
            print(f"Thresholds saved to {filepath}")
        except Exception as e:
            print(f"Error saving thresholds: {e}")
    
    def load_thresholds(self, filepath='user_thresholds.pkl'):
        """Load user thresholds from disk."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    self.user_thresholds = pickle.load(f)
                print(f"Loaded thresholds for {len(self.user_thresholds)} users")
            except Exception as e:
                print(f"Error loading thresholds: {e}")
        else:
            print(f"No saved thresholds found at {filepath}")


def calculate_confidence_score(distances, threshold):
    """
    Calculate a confidence score for the verification decision.
    
    Args:
        distances: array of cosine distances
        threshold: verification threshold
    
    Returns:
        confidence: float between 0 and 1
    """
    mean_distance = np.mean(distances)
    
    # How far from threshold?
    margin = abs(mean_distance - threshold)
    max_margin = 0.3
    
    confidence = min(margin / max_margin, 1.0)
    
    return confidence