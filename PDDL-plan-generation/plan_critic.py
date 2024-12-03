# pocketgroq/plan_critic.py

from typing import List, Dict, Any, Optional
import logging
import random
from dataclasses import dataclass
from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIError

logger = logging.getLogger(__name__)

@dataclass
class PDDLConstraint:
    negated: bool
    temporal_operator: str  # 'always', 'sometime', etc.
    predicate: str
    arguments: List[str]

    def to_pddl(self) -> str:
        """Convert the constraint to PDDL format."""
        pred = f"({self.predicate} {' '.join(self.arguments)})"
        temp = f"({self.temporal_operator} {pred})"
        return f"(not {temp})" if self.negated else temp

class GroqPlanCritic:
    def __init__(self, groq_provider: GroqProvider, 
                 population_size: int = 20,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7):
        self.groq = groq_provider
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.temporal_operators = ["always", "sometime", "at-most-once", "sometime-before"]
        self.current_population: List[List[PDDLConstraint]] = []

    def ground_preferences(self, preferences: List[str]) -> List[str]:
        """Convert natural language preferences to mid-level goals using GPT-4."""
        prompt = """Convert each planning preference to a mid-level goal that can be expressed in PDDL.
        For example:
        Preference: "Make sure the scout asset only visits the endpoint once"
        Mid-level goal: "Limit the scout asset to visiting the endpoint at most one time throughout the plan"
        
        Preferences to convert:
        {}
        """.format("\n".join(f"- {p}" for p in preferences))
        
        response = self.groq.generate(prompt=prompt)
        return [goal.strip() for goal in response.split("\n") if goal.strip()]

    def initialize_population(self, mid_level_goals: List[str], problem_file: str) -> List[List[PDDLConstraint]]:
        """Generate initial population of PDDL constraints."""
        prompt = """Given these mid-level planning goals and PDDL problem, generate valid PDDL constraints.
        Problem file:
        {}
        
        Mid-level goals:
        {}
        
        Generate PDDL constraints that would satisfy these goals.""".format(
            problem_file, "\n".join(f"- {g}" for g in mid_level_goals))
        
        base_constraints = self._parse_constraints(self.groq.generate(prompt=prompt))
        
        # Create initial population through mutation
        population = [base_constraints]
        while len(population) < self.population_size:
            mutated = self._mutate(base_constraints.copy())
            population.append(mutated)
            
        self.current_population = population
        return population

    def _parse_constraints(self, pddl_text: str) -> List[PDDLConstraint]:
        """Parse PDDL constraints from text into structured format."""
        constraints = []
        # Basic parsing of constraint structure
        for line in pddl_text.split("\n"):
            if not line.strip() or ";" in line:
                continue
                
            parts = line.strip("() ").split()
            if not parts:
                continue
                
            if parts[0] == "not":
                negated = True
                parts = parts[1:]
            else:
                negated = False
                
            if parts and parts[0] in self.temporal_operators:
                temporal_op = parts[0]
                predicate = parts[1]
                args = parts[2:]
                
                constraints.append(PDDLConstraint(
                    negated=negated,
                    temporal_operator=temporal_op,
                    predicate=predicate,
                    arguments=args
                ))
                
        return constraints

    def _mutate(self, constraints: List[PDDLConstraint]) -> List[PDDLConstraint]:
        """Mutate a set of constraints."""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["add", "remove", "modify"])
            
            if mutation_type == "add" and constraints:
                # Copy and modify an existing constraint
                base = random.choice(constraints)
                new = PDDLConstraint(
                    negated=not base.negated,
                    temporal_operator=random.choice(self.temporal_operators),
                    predicate=base.predicate,
                    arguments=base.arguments.copy()
                )
                constraints.append(new)
                
            elif mutation_type == "remove" and len(constraints) > 1:
                constraints.remove(random.choice(constraints))
                
            elif mutation_type == "modify" and constraints:
                constraint = random.choice(constraints)
                mod_type = random.choice(["negate", "temporal", "args"])
                
                if mod_type == "negate":
                    constraint.negated = not constraint.negated
                elif mod_type == "temporal":
                    constraint.temporal_operator = random.choice(self.temporal_operators)
                elif mod_type == "args":
                    # Shuffle arguments while maintaining validity
                    random.shuffle(constraint.arguments)
                    
        return constraints

    def _crossover(self, parent1: List[PDDLConstraint], parent2: List[PDDLConstraint]) -> List[PDDLConstraint]:
        """Perform crossover between two parent constraint sets."""
        if not parent1 or not parent2:
            return parent1 or parent2
            
        crossover_point = random.randint(0, min(len(parent1), len(parent2)))
        return parent1[:crossover_point] + parent2[crossover_point:]

    def evaluate_fitness(self, constraints: List[PDDLConstraint], feedback: List[str]) -> float:
        """Evaluate fitness of a constraint set against user feedback."""
        constraints_pddl = "\n".join(c.to_pddl() for c in constraints)
        
        prompt = f"""Given these PDDL constraints and user feedback, score how well the constraints satisfy the feedback from 0 to 1.
        
        PDDL Constraints:
        {constraints_pddl}
        
        User Feedback:
        {chr(10).join(feedback)}
        
        Score (0-1):"""
        
        try:
            score = float(self.groq.generate(prompt=prompt))
            return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
        except (ValueError, GroqAPIError) as e:
            logger.error(f"Error evaluating fitness: {e}")
            return 0.0

    def optimize(self, feedback: List[str], max_generations: int = 50) -> List[PDDLConstraint]:
        """Run genetic algorithm to optimize constraints."""
        for generation in range(max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = [(constraints, self.evaluate_fitness(constraints, feedback))
                            for constraints in self.current_population]
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Early stopping if we find a perfect solution
            if fitness_scores[0][1] >= 0.99:
                return fitness_scores[0][0]
            
            # Select parents for next generation
            parents = [pair[0] for pair in fitness_scores[:self.population_size//2]]
            
            # Create next generation through crossover and mutation
            next_generation = parents.copy()  # Keep best performing constraints
            
            while len(next_generation) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = random.choice(parents)
                    parent2 = random.choice(parents)
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._mutate(random.choice(parents).copy())
                next_generation.append(child)
            
            self.current_population = next_generation
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Best fitness: {fitness_scores[0][1]}")
        
        # Return best solution found
        return max(self.current_population,
                  key=lambda c: self.evaluate_fitness(c, feedback))

    def generate_plan(self, problem_file: str, preferences: List[str]) -> str:
        """Generate a plan that satisfies the given preferences."""
        # Ground natural language preferences to mid-level goals
        mid_level_goals = self.ground_preferences(preferences)
        
        # Initialize population
        self.initialize_population(mid_level_goals, problem_file)
        
        # Optimize constraints
        best_constraints = self.optimize(preferences)
        
        # Convert constraints to PDDL
        constraints_pddl = "\n".join(c.to_pddl() for c in best_constraints)
        
        return constraints_pddl