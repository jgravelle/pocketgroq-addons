from pocketgroq import GroqProvider
from plan_critic import GroqPlanCritic
import os
import logging
import sys
import time

# Set up logging with more verbose output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
    logger.debug("Starting test script...")
    
    # Check if GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        logger.error("GROQ_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Initialize GroqProvider (using API key from environment)
    try:
        logger.debug("Attempting to initialize GroqProvider...")
        groq = GroqProvider()
        logger.info("Successfully initialized GroqProvider")
    except Exception as e:
        logger.error(f"Failed to initialize GroqProvider: {str(e)}")
        sys.exit(1)

    try:
        # Create PlanCritic instance
        logger.debug("Creating PlanCritic instance...")
        plan_critic = GroqPlanCritic(groq)
        logger.info("Successfully created PlanCritic instance")

        # Example preferences from the paper
        preferences = [
            "Make sure the scout asset only visits the endpoint once",
            "We need to clear the route from debris station 0 to the endpoint within 5 hours",
            "Don't remove any underwater debris"
        ]

        logger.info("Testing GroqPlanCritic with waterway restoration problem...")

        # Save problem to temporary file
        problem_file = "waterway_problem.pddl"
        logger.debug(f"Writing problem to temporary file: {problem_file}")
        with open(problem_file, "w") as f:
            f.write(SAMPLE_PROBLEM)
        logger.info("Successfully wrote problem file")

        # Generate plan constraints
        logger.info("Generating plan constraints from preferences...")
        time.sleep(3)  # Add delay before API call
        constraints_pddl = plan_critic.generate_plan(problem_file, preferences)

        logger.info("\nGenerated PDDL Constraints:")
        print(constraints_pddl)

        # Test constraint grounding
        logger.info("\nTesting preference grounding...")
        time.sleep(3)  # Add delay before API call
        mid_level_goals = plan_critic.ground_preferences(preferences)
        
        logger.info("Mid-level goals:")
        for i, goal in enumerate(mid_level_goals, 1):
            print(f"{i}. {goal}")

        # Test population initialization
        logger.info("\nTesting population initialization...")
        time.sleep(3)  # Add delay before API call
        initial_pop = plan_critic.initialize_population(mid_level_goals, SAMPLE_PROBLEM)
        
        logger.info(f"Generated initial population of {len(initial_pop)} constraint sets")
        logger.info("First constraint set:")
        for constraint in initial_pop[0]:
            print(constraint.to_pddl())

        # Test fitness evaluation
        logger.info("\nTesting fitness evaluation...")
        time.sleep(3)  # Add delay before API call
        fitness = plan_critic.evaluate_fitness(initial_pop[0], preferences)
        logger.info(f"Fitness score of first constraint set: {fitness}")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(problem_file):
            try:
                os.remove(problem_file)
                logger.debug("Cleaned up temporary problem file")
            except Exception as e:
                logger.error(f"Failed to clean up problem file: {str(e)}")

if __name__ == "__main__":
    main()
