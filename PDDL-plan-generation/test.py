from pocketgroq import GroqProvider
from plan_critic import GroqPlanCritic
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample PDDL problem for waterway restoration domain
SAMPLE_PROBLEM = """
(define (problem waterway-restoration)
    (:domain waterway)
    (:objects
        sct_ast_0 - scout_asset
        deb_ast_0 - debris_asset
        slv_ast_0 - salvage_asset
        wpt_ini wpt_end wpt_a wpt_b_0 deb_stn_0 - waypoint
        u_deb_ini_b_0 u_deb_b_0_end - underwater_debris
        ship_0 - ship
    )
    (:init
        (scout_asset_at sct_ast_0 wpt_ini)
        (debris_asset_at deb_ast_0 deb_stn_0)
        (salvage_asset_at slv_ast_0 wpt_ini)
        (ship_at ship_0 wpt_b_0)
        (underwater_debris_at u_deb_ini_b_0 wpt_ini)
        (underwater_debris_at u_deb_b_0_end wpt_b_0)
        (blocked wpt_b_0 wpt_end)
    )
    (:goal
        (and
            (ship_at ship_0 wpt_end)
            (not (exists (?w1 ?w2 - waypoint) (blocked ?w1 ?w2)))
        )
    )
)
"""

def main():
    # Initialize GroqProvider (using API key from environment)
    try:
        groq = GroqProvider()
    except Exception as e:
        logger.error(f"Failed to initialize GroqProvider: {e}")
        return

    # Create PlanCritic instance
    plan_critic = GroqPlanCritic(groq)

    # Example preferences from the paper
    preferences = [
        "Make sure the scout asset only visits the endpoint once",
        "We need to clear the route from debris station 0 to the endpoint within 5 hours",
        "Don't remove any underwater debris"
    ]

    logger.info("Testing GroqPlanCritic with waterway restoration problem...")

    try:
        # Save problem to temporary file
        problem_file = "waterway_problem.pddl"
        with open(problem_file, "w") as f:
            f.write(SAMPLE_PROBLEM)

        # Generate plan constraints
        logger.info("Generating plan constraints from preferences...")
        constraints_pddl = plan_critic.generate_plan(problem_file, preferences)

        logger.info("\nGenerated PDDL Constraints:")
        print(constraints_pddl)

        # Test constraint grounding
        logger.info("\nTesting preference grounding...")
        mid_level_goals = plan_critic.ground_preferences(preferences)
        
        logger.info("Mid-level goals:")
        for i, goal in enumerate(mid_level_goals, 1):
            print(f"{i}. {goal}")

        # Test population initialization
        logger.info("\nTesting population initialization...")
        initial_pop = plan_critic.initialize_population(mid_level_goals, SAMPLE_PROBLEM)
        
        logger.info(f"Generated initial population of {len(initial_pop)} constraint sets")
        logger.info("First constraint set:")
        for constraint in initial_pop[0]:
            print(constraint.to_pddl())

        # Test fitness evaluation
        logger.info("\nTesting fitness evaluation...")
        fitness = plan_critic.evaluate_fitness(initial_pop[0], preferences)
        logger.info(f"Fitness score of first constraint set: {fitness}")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
    finally:
        # Cleanup
        if os.path.exists(problem_file):
            os.remove(problem_file)

if __name__ == "__main__":
    main()