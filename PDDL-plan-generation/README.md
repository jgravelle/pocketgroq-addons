# GroqPlanCritic

A PocketGroq add-on that implements feedback-driven PDDL plan generation based on the [PlanCritic paper](https://arxiv.org/abs/2412.00300). GroqPlanCritic allows you to generate and optimize PDDL plans using natural language preferences and LLM-based feedback.

## Features

- Natural language preference handling
- Genetic algorithm-based plan optimization
- LLM-powered reward model for constraint evaluation
- Integration with PocketGroq's existing infrastructure
- Support for temporal and state trajectory constraints
- Built-in preference grounding to symbolic goals

## Installation

GroqPlanCritic is included as part of PocketGroq. Ensure you have PocketGroq installed with all dependencies:

```bash
pip install pocketgroq
```

Make sure you have set your Groq API key:

```bash
export GROQ_API_KEY='your-api-key'
```

## Quick Start

```python
from pocketgroq import GroqProvider
from pocketgroq.plan_critic import GroqPlanCritic

# Initialize providers
groq = GroqProvider()
plan_critic = GroqPlanCritic(groq)

# Define preferences and problem
preferences = [
    "Make sure the scout asset only visits the endpoint once",
    "Clear the route within 5 hours",
    "Don't remove any underwater debris"
]

problem_file = "path/to/your/problem.pddl"

# Generate optimized plan constraints
constraints = plan_critic.generate_plan(problem_file, preferences)
print(constraints)
```

## Core Components

### PDDLConstraint

Represents a single PDDL constraint with properties:
- `negated`: Boolean indicating if constraint is negated
- `temporal_operator`: Temporal operator (always, sometime, etc.)
- `predicate`: The PDDL predicate
- `arguments`: List of predicate arguments

### GroqPlanCritic

Main class providing:
- `ground_preferences()`: Convert natural language to mid-level goals
- `initialize_population()`: Create initial constraint population
- `optimize()`: Run genetic algorithm optimization
- `generate_plan()`: Generate complete plan with constraints

## Configuration

You can customize the genetic algorithm parameters:

```python
critic = GroqPlanCritic(
    groq_provider,
    population_size=20,    # Size of constraint population
    mutation_rate=0.3,     # Probability of mutation
    crossover_rate=0.7     # Probability of crossover
)
```

## Examples

### Basic Usage

```python
from pocketgroq import GroqProvider
from pocketgroq.plan_critic import GroqPlanCritic

groq = GroqProvider()
critic = GroqPlanCritic(groq)

# Simple preference example
preferences = ["Deliver package A before package B"]
constraints = critic.generate_plan("delivery.pddl", preferences)
```

### Custom Optimization

```python
# Ground preferences to mid-level goals
goals = critic.ground_preferences(preferences)

# Initialize population
population = critic.initialize_population(goals, problem_file)

# Run optimization with custom generations
best_constraints = critic.optimize(preferences, max_generations=100)
```

## Advanced Features

### Constraint Validation

The system automatically validates generated constraints against the PDDL problem specification.

### Fitness Evaluation

The LLM-based reward model evaluates constraints by:
1. Analyzing adherence to user preferences
2. Checking constraint consistency
3. Validating temporal logic
4. Assessing completeness

### Genetic Operations

Available mutation operations:
- Add new constraints
- Remove existing constraints
- Modify constraint parameters
- Change temporal operators
- Negate constraints

Crossover operations:
- Single-point crossover
- Constraint set merging
- Preservation of valid subsets

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## How It Works

GroqPlanCritic follows the architecture outlined in the PlanCritic paper:

1. **Preference Grounding**: Natural language preferences are converted to symbolic mid-level goals using GPT-4

2. **Population Initialization**: Creates initial population of PDDL constraints based on mid-level goals

3. **Genetic Optimization**: 
   - Mutation and crossover operations generate new constraint variations
   - LLM-based reward model evaluates fitness
   - Best constraints are selected for next generation

4. **Plan Generation**: Final constraints are used to generate an optimized plan

## Citation

If you use GroqPlanCritic in your research, please cite both the original PlanCritic paper and PocketGroq:

```bibtex
@article{burns2024plancritic,
  title={PlanCritic: Formal Planning with Human Feedback},
  author={Burns, Owen and Hughes, Dana and Sycara, Katia},
  journal={arXiv preprint arXiv:2412.00300},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the PlanCritic paper by Burns et al.
- Uses the Groq API and PocketGroq library
- Built on PDDL planning capabilities