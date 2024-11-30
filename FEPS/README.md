# FEPS Enhancement for PocketGroq

An extension to PocketGroq that implements Free Energy Projective Simulation (FEPS) for improved environmental understanding and decision-making under uncertainty.  Inspired by this whitepaper:  https://arxiv.org/abs/2411.14991

## Overview

FEPS Enhancement adds interpretable decision-making capabilities to PocketGroq using simple rules to simulate intelligent behavior without relying on complex mathematics or deep learning. The enhancement enables agents to:

- Build accurate mental models of their environment based on internal rewards
- Handle partially observable environments with ambiguous states
- Make optimal decisions even with incomplete information
- Learn from prediction accuracy rather than external rewards

## Key Features

### Clone-Based Memory Structure
- **Clone Clips**: Multiple representations of the same observation to handle ambiguous situations
- **Belief States**: Maintains multiple hypotheses about the current state
- **Confidence Tracking**: Accumulates confidence values along successful prediction trajectories
- **Internal Reward System**: Based on prediction accuracy rather than external rewards

### Enhanced Decision Making
- Uncertainty estimation for each possible action
- Belief state tracking and filtering
- Prediction-based learning and adaptation
- Flexible action selection based on uncertainty levels

## Installation

Add the FEPS enhancement to your existing PocketGroq installation:

```bash
pip install pocketgroq
# Clone the FEPS enhancement
git clone https://github.com/yourusername/pocketgroq-feps.git
cd pocketgroq-feps
pip install -e .
```

## Quick Start

```python
from pocketgroq import GroqProvider
from pocketgroq_feps import enhance_groq_provider

# Initialize providers
groq_provider = GroqProvider(api_key="your-api-key")
observations = ["state1", "state2", "state3"]
enhanced_provider = enhance_groq_provider(groq_provider, observations)

# Process observations
enhanced_provider.process_observation("state1")
prediction = enhanced_provider.get_prediction("action1")
beliefs = enhanced_provider.get_belief_states()
uncertainty = enhanced_provider.evaluate_uncertainty("action1")
```

## Core Components

### FEPSMemory
Manages the episodic and compositional memory (ECM) with features like:
- Clone clip creation and management
- Belief state tracking
- Confidence-based learning
- Trajectory tracking for successful predictions

### FEPSEnhancedGroqProvider
Wraps the standard GroqProvider with FEPS capabilities:
- Prediction generation
- Uncertainty evaluation
- Belief state management
- Model updates based on prediction accuracy

## Usage Examples

### Basic Navigation Task
```python
# Initialize environment and provider
env = GridWorld()
enhanced_groq = enhance_groq_provider(groq_provider, env.get_observations())

# Process observation and get prediction
current_obs = env.get_observation()
enhanced_groq.process_observation(current_obs)
prediction = enhanced_groq.get_prediction("move_right")

# Check uncertainty
uncertainty = enhanced_groq.evaluate_uncertainty("move_right")
```

### Handling Ambiguous States
```python
# Process observation with multiple possible interpretations
enhanced_groq.process_observation("ambiguous_state")

# Get current belief states
beliefs = enhanced_groq.get_belief_states()
print(f"Possible interpretations: {beliefs}")

# Make prediction considering all beliefs
prediction = enhanced_groq.get_prediction("action")
```

## Integration with Existing PocketGroq Features

### RAG Integration
FEPS enhances RAG capabilities by:
- Using belief states to improve document retrieval relevance
- Incorporating prediction accuracy into index updates
- Providing uncertainty estimates for retrieval confidence

### Chain of Thought Integration
Works seamlessly with existing CoT features:
- Enhanced reasoning with belief state information
- Improved uncertainty handling in complex chains
- More reliable validity checks using prediction accuracy

## Testing

Run the included test suite to verify the FEPS enhancement:

```bash
python -m pytest tests/test_feps.py
```

Or try the example script:

```bash
python examples/feps_test.py
```

## Parameters and Configuration

### Key Parameters
- `num_clones_per_observation`: Number of clone clips per observation (default: 2)
- `gamma`: Forgetting parameter for h-value updates (default: 0.1)
- `base_reward`: Base reward for correct predictions (default: 1.0)

### Configuration Example
```python
enhanced_groq = FEPSEnhancedGroqProvider(
    groq_provider,
    num_clones_per_observation=3,
    gamma=0.15
)
```

## How It Works

1. **Observation Processing**
   - Creates multiple clone clips for each observation
   - Maintains active belief states
   - Updates confidence based on prediction accuracy

2. **Prediction Generation**
   - Samples next states based on h-values
   - Considers all active belief states
   - Returns most likely prediction

3. **Model Updates**
   - Tracks successful prediction trajectories
   - Distributes rewards based on confidence
   - Updates h-values with forgetting mechanism

4. **Uncertainty Evaluation**
   - Calculates entropy of predictions
   - Considers all possible next states
   - Provides uncertainty measure for actions

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this enhancement in your research, please cite:

```bibtex
@misc{pocketgroq-feps,
  author = J. Gravelle,
  title = {FEPS Enhancement for PocketGroq},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jgravelle/pocketgroq-addons/FEPS}
}
```

## References

1. Original FEPS paper: "Free Energy Projective Simulation (FEPS): Active inference with interpretability"
2. PocketGroq documentation
3. Active Inference literature