import unittest
from typing import List, Tuple, Dict
from pocketgroq import GroqProvider
from dataclasses import dataclass
import numpy as np

@dataclass
class GridWorld:
    """Simple 3x3 grid world environment for testing FEPS"""
    # Grid layout:
    # [0,1,2]
    # [3,4,5]
    # [6,7,8]
    
    def __init__(self):
        self.position = 4  # Start in center
        self.size = 3
        self.actions = ["up", "down", "left", "right"]
        
        # Define observations as "distance from center"
        self.observations = {
            0: "corner", 1: "edge", 2: "corner",
            3: "edge", 4: "center", 5: "edge",
            6: "corner", 7: "edge", 8: "corner"
        }
    
    def get_observation(self) -> str:
        """Get current observation based on position"""
        return self.observations[self.position]
    
    def step(self, action: str) -> Tuple[str, bool]:
        """Take action and return new observation and whether valid"""
        old_pos = self.position
        
        # Calculate new position
        if action == "up" and self.position >= self.size:
            self.position -= self.size
        elif action == "down" and self.position < self.size * (self.size - 1):
            self.position += self.size
        elif action == "left" and self.position % self.size != 0:
            self.position -= 1
        elif action == "right" and (self.position + 1) % self.size != 0:
            self.position += 1
            
        valid_move = old_pos != self.position
        return self.get_observation(), valid_move

def test_feps():
    # Initialize environment and FEPS-enhanced provider
    env = GridWorld()
    groq = GroqProvider(api_key="test_key")  # Mock API key for testing
    
    # Get unique observations from environment
    observations = list(set(env.observations.values()))
    enhanced_groq = enhance_groq_provider(groq, observations)
    
    # Training loop parameters
    num_episodes = 100
    max_steps = 20
    total_correct_predictions = 0
    total_predictions = 0

    print("\nStarting FEPS training...")
    
    for episode in range(num_episodes):
        env.position = 4  # Reset to center
        current_obs = env.get_observation()
        enhanced_groq.process_observation(current_obs)
        
        for step in range(max_steps):
            # Choose random action to explore
            action = np.random.choice(env.actions)
            
            # Get FEPS prediction
            prediction = enhanced_groq.get_prediction(action)
            
            # Take action in environment
            next_obs, valid = env.step(action)
            
            # Process new observation and update model
            enhanced_groq.process_observation(next_obs, action)
            
            # Track prediction accuracy
            if prediction is not None:
                total_predictions += 1
                if prediction == next_obs:
                    total_correct_predictions += 1
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            accuracy = (total_correct_predictions / total_predictions) if total_predictions > 0 else 0
            print(f"Episode {episode + 1}, Prediction Accuracy: {accuracy:.2%}")
            
            # Show current belief states
            beliefs = enhanced_groq.get_belief_states()
            print(f"Current belief states: {beliefs}")
            
            # Show uncertainty for each action
            uncertainties = {
                action: enhanced_groq.evaluate_uncertainty(action)
                for action in env.actions
            }
            print("Action uncertainties:", {a: f"{u:.2f}" for a, u in uncertainties.items()})
            print()

    print("\nTraining complete!")
    print(f"Final prediction accuracy: {(total_correct_predictions / total_predictions):.2%}")
    
    # Test prediction in specific scenarios
    print("\nTesting specific scenarios:")
    
    # Test corner prediction
    env.position = 4  # Center
    enhanced_groq.process_observation(env.get_observation())
    
    # Predict moving to corner
    corner_prediction = enhanced_groq.get_prediction("up")
    print(f"Prediction for moving up from center: {corner_prediction}")
    
    # Show belief states and uncertainty
    print(f"Belief states: {enhanced_groq.get_belief_states()}")
    print(f"Uncertainty for 'up' action: {enhanced_groq.evaluate_uncertainty('up'):.2f}")

if __name__ == "__main__":
    test_feps()