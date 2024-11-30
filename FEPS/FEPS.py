from typing import List, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ClipNode:
    """Represents a node in the episodic and compositional memory (ECM)"""
    id: str
    observation: str  # The sensory observation this clip is associated with
    h_values: Dict[str, float] = defaultdict(float)  # Edge weights to other clips
    confidence: float = 0.0

class FEPSMemory:
    """Implements the FEPS memory structure with clone clips and belief states"""
    def __init__(self, num_clones_per_observation: int = 2):
        self.num_clones = num_clones_per_observation
        self.clips: Dict[str, ClipNode] = {}
        self.current_beliefs: Set[str] = set()  # Currently active belief states
        self.trajectory: List[str] = []  # Current sequence of correct predictions
        self.gamma = 0.1  # Forgetting parameter
        
    def initialize_clips(self, observations: List[str]):
        """Create clone clips for each observation"""
        for obs in observations:
            for i in range(self.num_clones):
                clip_id = f"{obs}_clone_{i}"
                self.clips[clip_id] = ClipNode(id=clip_id, observation=obs)
    
    def update_beliefs(self, observation: str):
        """Update belief states based on new observation"""
        if not self.current_beliefs:
            # Initialize with all compatible clips if no current beliefs
            self.current_beliefs = {
                clip_id for clip_id, clip in self.clips.items() 
                if clip.observation == observation
            }
        else:
            # Filter beliefs to keep only those compatible with observation
            self.current_beliefs = {
                belief for belief in self.current_beliefs
                if self.clips[belief].observation == observation
            }
            
            if not self.current_beliefs:
                # Reset if all beliefs eliminated
                self.current_beliefs = {
                    clip_id for clip_id, clip in self.clips.items() 
                    if clip.observation == observation
                }
                self.trajectory = []

    def predict_next_observation(self, action: str) -> str:
        """Make prediction about next observation given current beliefs and action"""
        if not self.current_beliefs:
            return None
            
        # Sample next belief state for each current belief
        predictions = []
        for belief in self.current_beliefs:
            clip = self.clips[belief]
            # Sample next clip based on h-values
            next_clip_id = self._sample_next_clip(clip, action)
            if next_clip_id:
                predictions.append(self.clips[next_clip_id].observation)
                
        # Return most common prediction
        if predictions:
            return max(set(predictions), key=predictions.count)
        return None

    def update_model(self, 
                    current_observation: str,
                    action: str, 
                    next_observation: str,
                    correct_prediction: bool):
        """Update the model based on prediction accuracy"""
        self.update_beliefs(current_observation)
        
        if correct_prediction:
            # Extend trajectory
            self.trajectory.extend([b for b in self.current_beliefs])
            
            # Update confidence for edges in trajectory
            for i in range(len(self.trajectory) - 1):
                src_clip = self.clips[self.trajectory[i]]
                dst_clip = self.clips[self.trajectory[i + 1]]
                edge_key = f"{action}_{dst_clip.id}"
                src_clip.confidence += 1
                
        else:
            # Distribute rewards based on confidence
            self._distribute_rewards()
            self.trajectory = []
            
        # Update beliefs for next observation
        self.update_beliefs(next_observation)

    def _sample_next_clip(self, clip: ClipNode, action: str) -> Optional[str]:
        """Sample next clip based on h-values"""
        possible_next = [
            (next_id, clip.h_values[f"{action}_{next_id}"])
            for next_id in self.clips.keys()
        ]
        if not possible_next:
            return None
            
        # Convert to probabilities
        h_values = np.array([h for _, h in possible_next])
        if h_values.sum() == 0:
            # Uniform if no learning yet
            probs = np.ones(len(possible_next)) / len(possible_next)
        else:
            probs = h_values / h_values.sum()
            
        # Sample next clip
        next_idx = np.random.choice(len(possible_next), p=probs)
        return possible_next[next_idx][0]

    def _distribute_rewards(self, base_reward: float = 1.0):
        """Distribute rewards to edges based on confidence"""
        if len(self.trajectory) < 2:
            return
            
        for i in range(len(self.trajectory) - 1):
            src_clip = self.clips[self.trajectory[i]]
            dst_clip = self.clips[self.trajectory[i + 1]]
            
            # Update h-value with confidence-weighted reward
            edge_key = f"{action}_{dst_clip.id}"
            old_h = src_clip.h_values[edge_key]
            reward = base_reward * src_clip.confidence
            
            # Apply forgetting and reward
            src_clip.h_values[edge_key] = (
                old_h - self.gamma * (old_h - 1.0) + reward
            )
            
        # Reset confidence values
        for clip_id in self.trajectory:
            self.clips[clip_id].confidence = 0.0

class FEPSEnhancedGroqProvider:
    """Enhances GroqProvider with FEPS-based memory and learning"""
    def __init__(self, groq_provider, num_clones_per_observation: int = 2):
        self.groq = groq_provider
        self.feps_memory = FEPSMemory(num_clones_per_observation)
        self.observation_history = []
        
    def initialize(self, observations: List[str]):
        """Initialize FEPS memory with possible observations"""
        self.feps_memory.initialize_clips(observations)
        
    def process_observation(self, observation: str, action: str = None):
        """Process new observation and update model"""
        self.observation_history.append(observation)
        
        if len(self.observation_history) > 1 and action:
            # Check if previous prediction was correct
            predicted = self.feps_memory.predict_next_observation(action)
            correct = predicted == observation if predicted else False
            
            # Update model
            self.feps_memory.update_model(
                self.observation_history[-2],
                action,
                observation,
                correct
            )
            
    def get_belief_states(self) -> Set[str]:
        """Get current belief states"""
        return self.feps_memory.current_beliefs
        
    def get_prediction(self, action: str) -> str:
        """Get prediction for next observation given action"""
        return self.feps_memory.predict_next_observation(action)
        
    def evaluate_uncertainty(self, action: str) -> float:
        """Evaluate uncertainty of predictions for an action"""
        predictions = []
        for belief in self.feps_memory.current_beliefs:
            clip = self.feps_memory.clips[belief]
            next_clips = [
                k.split('_', 1)[1] for k, v in clip.h_values.items()
                if k.startswith(f"{action}_") and v > 0
            ]
            predictions.extend(
                self.feps_memory.clips[next_id].observation 
                for next_id in next_clips
            )
        
        if not predictions:
            return 1.0
            
        # Calculate entropy of predictions
        counts = defaultdict(int)
        for p in predictions:
            counts[p] += 1
        probs = [c/len(predictions) for c in counts.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy

def enhance_groq_provider(groq_provider, observations: List[str]):
    """Factory function to create FEPS-enhanced GroqProvider"""
    enhanced = FEPSEnhancedGroqProvider(groq_provider)
    enhanced.initialize(observations)
    return enhanced