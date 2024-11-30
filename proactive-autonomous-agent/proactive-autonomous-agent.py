from typing import Dict, Any, List, Optional, Generator
from pocketgroq.autonomous_agent import AutonomousAgent
from pocketgroq import GroqProvider

class ProactiveAutonomousAgent(AutonomousAgent):
    """
    Enhanced version of AutonomousAgent with proactive capabilities based on the ProactiveAgent methodology.
    Extends the base AutonomousAgent class with environmental awareness and proactive task prediction.
    """
    def __init__(
        self, 
        groq_provider: GroqProvider, 
        max_sources: int = 5, 
        search_delay: float = 2.0,
        model: str = "llama3-8b-8192",
        temperature: float = 0.0,
        proactive_threshold: float = 0.7
    ):
        super().__init__(groq_provider, max_sources, search_delay, model, temperature)
        self.proactive_threshold = proactive_threshold
        self.environment_state = {
            "events": [],
            "activities": [],
            "state": {}
        }

    def update_environment(self, event: Dict[str, Any]) -> None:
        """
        Updates the agent's understanding of the environment based on new events.
        
        Args:
            event (Dict[str, Any]): New event information
        """
        self.environment_state["events"].append(event)
        
        # Maintain a rolling window of recent events
        if len(self.environment_state["events"]) > 100:
            self.environment_state["events"] = self.environment_state["events"][-100:]

    def detect_user_needs(self, recent_events: List[Dict[str, Any]]) -> Optional[str]:
        """
        Analyzes recent events to detect potential user needs using Chain-of-Thought reasoning.
        
        Args:
            recent_events (List[Dict[str, Any]]): Recent events to analyze
            
        Returns:
            Optional[str]: Detected need if any, None otherwise
        """
        prompt = f"""
        Analyze the following recent user events and determine if there are any potential needs or tasks:

        Events:
        {recent_events}

        Using chain-of-thought reasoning:
        1. What is the user's current context and goal?
        2. Are there any patterns indicating potential needs?
        3. What assistance would be most valuable?

        Respond in JSON format:
        {{
            "detected_need": "description or null if none detected",
            "confidence": "float between 0 and 1",
            "reasoning": "step by step explanation"
        }}
        """

        response = self.groq.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            json_mode=True
        )
        
        try:
            result = eval(response)  # Safe since we control the input
            if result["confidence"] >= self.proactive_threshold:
                return result["detected_need"]
        except:
            return None
        
        return None

    def draft_proactive_task(self, detected_need: str) -> Dict[str, Any]:
        """
        Creates a draft task proposal based on detected need.
        
        Args:
            detected_need (str): The detected user need
            
        Returns:
            Dict[str, Any]: Draft task proposal
        """
        prompt = f"""
        Create a task proposal for the following detected need:
        {detected_need}

        The proposal should be:
        1. Specific and actionable
        2. Minimally disruptive
        3. Clearly valuable to the user

        Respond in JSON format:
        {{
            "task": "specific task description",
            "value_proposition": "why this would help the user",
            "execution_plan": "how the task would be completed"
        }}
        """

        response = self.groq.generate(
            prompt=prompt,
            model=self.model, 
            temperature=self.temperature,
            json_mode=True
        )
        
        return eval(response)  # Safe since we control the input

    async def process_request_proactively(
        self, 
        request: str, 
        max_sources: int = None
    ) -> Generator[Dict[str, str], None, None]:
        """
        Enhanced version of process_request that includes proactive capabilities.
        
        Args:
            request (str): Initial request or context
            max_sources (int, optional): Maximum number of sources to consult
            
        Yields:
            Generator[Dict[str, str], None, None]: Status updates and results
        """
        # Update environment with the new request
        self.update_environment({"type": "request", "content": request})

        # First try standard processing
        async for result in super().process_request(request, max_sources):
            yield result

        # Then check for proactive opportunities
        recent_events = self.environment_state["events"][-10:]  # Last 10 events
        detected_need = self.detect_user_needs(recent_events)

        if detected_need:
            task_proposal = self.draft_proactive_task(detected_need)
            
            # Evaluate if the task is appropriate using reward model
            is_appropriate = self.groq.evaluate_response(
                request=f"Context: {recent_events}\nProposed task: {task_proposal}",
                response=task_proposal["task"]
            )

            if is_appropriate:
                yield {
                    "type": "proactive_suggestion",
                    "content": f"""
                    I noticed you might benefit from: {task_proposal['task']}
                    
                    This would help because: {task_proposal['value_proposition']}
                    
                    Would you like me to proceed?
                    """
                }

    def process_proactive_feedback(
        self, 
        feedback: str, 
        task_proposal: Dict[str, Any]
    ) -> None:
        """
        Processes user feedback on proactive suggestions to improve future predictions.
        
        Args:
            feedback (str): User's feedback
            task_proposal (Dict[str, Any]): The original task proposal
        """
        prompt = f"""
        Analyze the user's feedback on the proactive suggestion:
        
        Original proposal: {task_proposal}
        User feedback: {feedback}
        
        Extract key learning points to improve future suggestions.
        Respond in JSON format:
        {{
            "was_helpful": "boolean",
            "timing_appropriate": "boolean",
            "learning_points": ["list", "of", "learnings"]
        }}
        """
        
        response = self.groq.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            json_mode=True
        )
        
        # Store learnings for future improvements
        learnings = eval(response)  # Safe since we control the input
        self.environment_state["learnings"] = self.environment_state.get("learnings", []) + [learnings]