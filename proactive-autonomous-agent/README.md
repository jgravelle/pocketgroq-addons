# PocketGroq ProActive Agent

An extension to PocketGroq that adds proactive capabilities based on the research presented in ["Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance"](https://arxiv.org/abs/2410.12361) by Lu et al. (2024).

## Overview

This extension enhances PocketGroq's AutonomousAgent with proactive capabilities, allowing it to anticipate and initiate tasks without explicit human instructions. The implementation follows the methodology described in the original research paper, incorporating:

- Environmental awareness and context tracking
- Chain-of-thought reasoning for need detection
- Reward model-based evaluation of task proposals
- Feedback-driven learning

## Academic Attribution

This implementation is based on the methodology described in:

```bibtex
@article{lu2024proactive,
  title={Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance},
  author={Lu, Yaxi and Yang, Shenzhi and Qian, Cheng and Chen, Guirong and Luo, Qinyu and Wu, Yesai and Wang, Huadong and Cong, Xin and Zhang, Zhong and Lin, Yankai and others},
  journal={arXiv preprint arXiv:2410.12361},
  year={2024}
}
```

## Installation

```bash
pip install pocketgroq
```

Clone this repository and add the proactive agent extension:

```bash
git clone [repository-url]
cd pocketgroq-proactive
python setup.py install
```

## Usage

### Basic Usage

```python
from pocketgroq import GroqProvider
from proactive_agent import ProactiveAutonomousAgent

# Initialize the provider and agent
groq = GroqProvider(api_key="your-api-key")
agent = ProactiveAutonomousAgent(
    groq_provider=groq,
    max_sources=5,
    proactive_threshold=0.7
)

# Process requests with proactive capabilities
async for result in agent.process_request_proactively("Help me with Python programming"):
    if result["type"] == "proactive_suggestion":
        print("Proactive suggestion:", result["content"])
    else:
        print("Regular response:", result["content"])
```

### Configuration

The ProactiveAutonomousAgent accepts several configuration parameters:

```python
ProactiveAutonomousAgent(
    groq_provider,                # Required: GroqProvider instance
    max_sources=5,               # Maximum number of sources to consult
    search_delay=2.0,            # Delay between searches
    model="llama3-8b-8192",      # Model to use for generations
    temperature=0.0,             # Temperature for generations
    proactive_threshold=0.7      # Confidence threshold for proactive suggestions
)
```

## Key Features

### Environmental Awareness

The agent maintains awareness of:
- Events (Et): User actions and system events
- Activities (At): User's ongoing tasks and interactions
- State (St): Current environment configuration

### Proactive Task Detection

Uses chain-of-thought reasoning to:
1. Analyze recent events
2. Detect potential user needs
3. Draft appropriate task proposals
4. Evaluate proposal appropriateness

### Feedback Learning

Processes user feedback to improve:
- Task timing
- Suggestion relevance
- Intervention appropriateness

## Testing

Run the test suite:

```bash
pip install pytest pytest-asyncio
pytest tests/test_proactive_agent.py -v
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Special thanks to:
- The authors of the ProactiveAgent paper for their groundbreaking research
- The PocketGroq team for the excellent base implementation
- The open source community for valuable feedback and contributions

## Citing

If you use this implementation in your research, please cite J. Gravelle, this repository, and the original paper:

```bibtex
@article{lu2024proactive,
  title={Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance},
  author={Lu, Yaxi and Yang, Shenzhi and Qian, Cheng and Chen, Guirong and Luo, Qinyu and Wu, Yesai and Wang, Huadong and Cong, Xin and Zhang, Zhong and Lin, Yankai and others},
  journal={arXiv preprint arXiv:2410.12361},
  year={2024}
}
```