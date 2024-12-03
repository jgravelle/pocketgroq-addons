# PocketGroq Add-ons

A collection of advanced add-ons for PocketGroq, implementing cutting-edge AI research papers to enhance PocketGroq's capabilities. Each add-on is designed to provide specific functionality while maintaining seamless integration with the core PocketGroq framework.

## Components

### 1. FEPS (Free Energy Projective Simulation)
An enhancement that implements interpretable decision-making capabilities using Free Energy Projective Simulation. Based on research from [arXiv:2411.14991](https://arxiv.org/abs/2411.14991).

Key features:
- Clone-based memory structure for handling ambiguous situations
- Enhanced decision making with uncertainty estimation
- Prediction-based learning and adaptation
- Seamless integration with existing PocketGroq features

[Learn more about FEPS](./FEPS/README.md)

### 2. LongKey (Keyphrase Extraction)
A powerful keyphrase extraction add-on designed specifically for analyzing long documents with advanced contextual understanding. Implements methodology from [arXiv:2411.17863](https://arxiv.org/pdf/2411.17863).

Key features:
- Process documents up to 96K tokens
- Advanced embedding strategy using CNNs and max pooling
- Context-aware processing through Longformer architecture
- Flexible integration with PocketGroq

[Learn more about LongKey](./keyphrase-extraction/README.md)

### 3. Literature Review Generator
Extends PocketGroq with automated literature review generation capabilities, based on research from [arXiv:2411.18583](https://arxiv.org/abs/2411.18583).

Key features:
- PDF processing and metadata extraction
- Multi-document analysis
- Quality evaluation
- Academic writing standards preservation
- RAG integration

[Learn more about Literature Review Generator](./literature-review/README.md)

### 4. ProActive Agent
Enhances PocketGroq's AutonomousAgent with proactive capabilities based on research from [arXiv:2410.12361](https://arxiv.org/abs/2410.12361).

Key features:
- Environmental awareness and context tracking
- Chain-of-thought reasoning for need detection
- Reward model-based evaluation
- Feedback-driven learning

[Learn more about ProActive Agent](./proactive-autonomous-agent/README.md)

### 5. PDDL Plan Generation
A feedback-driven PDDL plan generation system based on research from [arXiv:2412.00300](https://arxiv.org/abs/2412.00300). Enables generation and optimization of PDDL plans using natural language preferences and LLM-based feedback.

Key features:
- Natural language preference handling
- Genetic algorithm-based plan optimization
- LLM-powered reward model for constraint evaluation
- Support for temporal and state trajectory constraints
- Built-in preference grounding to symbolic goals

[Learn more about PDDL Plan Generation](./PDDL-plan-generation/README.md)

## Installation

First, install PocketGroq:

```bash
pip install pocketgroq
```

Then clone this repository to get the add-ons:

```bash
git clone https://github.com/jgravelle/pocketgroq-addons.git
cd pocketgroq-addons
```

Each add-on can be installed manually from its respective directory:

```bash
# Install FEPS
cd FEPS
pip install -e .

# Install LongKey
cd ../keyphrase-extraction
pip install -e .

# Install Literature Review Generator
cd ../literature-review
pip install -e .

# Install ProActive Agent
cd ../proactive-autonomous-agent
pip install -e .

# Install PDDL Plan Generation
cd ../PDDL-plan-generation
pip install -e .
```

## Quick Start

Here's a simple example combining multiple add-ons:

```python
from pocketgroq import GroqProvider
from pocketgroq_feps import enhance_groq_provider
from pocketgroq_longkey import add_longkey_to_groq
from literature_review import LiteratureReviewGenerator
from proactive_agent import ProactiveAutonomousAgent
from pocketgroq.plan_critic import GroqPlanCritic

# Initialize PocketGroq
groq = GroqProvider(api_key="your-api-key")

# Add desired enhancements
enhanced_groq = enhance_groq_provider(groq)
add_longkey_to_groq(enhanced_groq)

# Create specialized components
lit_review = LiteratureReviewGenerator(enhanced_groq)
proactive_agent = ProactiveAutonomousAgent(enhanced_groq)
plan_critic = GroqPlanCritic(enhanced_groq)

# Use enhanced capabilities
keyphrases = enhanced_groq.extract_keyphrases(text)
review = lit_review.generate_review(pdf_paths)
async for result in proactive_agent.process_request_proactively(query):
    print(result)
constraints = plan_critic.generate_plan(problem_file, preferences)
```

## Requirements

- Python >= 3.7
- PocketGroq >= 0.5.5
- PyTorch >= 1.7.0
- Transformers >= 4.0.0

Additional requirements may vary by add-on. Please check individual component READMEs for specific requirements.

## Contributing

We welcome contributions to any of the add-ons! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## Testing

Each add-on includes its own test suite. Run all tests with:

```bash
python -m pytest tests/
```

Or test specific components:

```bash
python -m pytest FEPS/test.py
python -m pytest keyphrase-extraction/test.py
python -m pytest literature-review/test.py
python -m pytest proactive-autonomous-agent/tests.py
python -m pytest PDDL-plan-generation/test.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use these add-ons in your research, please cite J. Gravelle, this repository, and the relevant research papers:

```bibtex
@misc{pocketgroq-addons,
  author = {J. Gravelle},
  title = {PocketGroq Add-ons},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jgravelle/pocketgroq-addons}
}
```

See individual component READMEs for specific paper citations.

## Acknowledgments

- Authors of the original research papers
- PocketGroq team for the excellent base framework
- Open source community for valuable feedback and contributions
