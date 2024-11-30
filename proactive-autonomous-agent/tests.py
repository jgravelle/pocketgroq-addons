import asyncio
import os
import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from pocketgroq import GroqProvider
from pocketgroq.exceptions import GroqAPIKeyMissingError
from proactive_agent import ProactiveAutonomousAgent

# Sample test data
SAMPLE_EVENTS = [
    {
        "time": "2024-11-29T10:00:00",
        "event": "User opens browser and navigates to Python documentation"
    },
    {
        "time": "2024-11-29T10:01:00",
        "event": "User searches for 'python async await examples'"
    },
    {
        "time": "2024-11-29T10:02:00",
        "event": "User opens VS Code and creates new file 'async_test.py'"
    }
]

SAMPLE_TASK_PROPOSAL = {
    "task": "Help set up an async/await code template",
    "value_proposition": "Save time with a working example structure",
    "execution_plan": "Provide a basic async/await template with comments"
}

@pytest.fixture
def mock_groq_provider():
    """Creates a mock GroqProvider for testing."""
    mock_provider = Mock(spec=GroqProvider)
    
    # Mock generate method
    def mock_generate(*args, **kwargs):
        if "json_mode" in kwargs and kwargs["json_mode"]:
            if "detected_need" in kwargs.get("prompt", ""):
                return json.dumps({
                    "detected_need": "async/await template assistance",
                    "confidence": 0.85,
                    "reasoning": "User is researching async/await and starting new file"
                })
            elif "task proposal" in kwargs.get("prompt", ""):
                return json.dumps(SAMPLE_TASK_PROPOSAL)
        return "Mock response"
    
    mock_provider.generate = Mock(side_effect=mock_generate)
    mock_provider.evaluate_response = Mock(return_value=True)
    return mock_provider

@pytest.fixture
def proactive_agent(mock_groq_provider):
    """Creates a ProactiveAutonomousAgent instance for testing."""
    return ProactiveAutonomousAgent(
        groq_provider=mock_groq_provider,
        max_sources=3,
        search_delay=0.1,
        proactive_threshold=0.7
    )

def test_proactive_agent_initialization(proactive_agent):
    """Tests the initialization of ProactiveAutonomousAgent."""
    assert proactive_agent.proactive_threshold == 0.7
    assert isinstance(proactive_agent.environment_state, dict)
    assert all(key in proactive_agent.environment_state 
              for key in ["events", "activities", "state"])

def test_update_environment(proactive_agent):
    """Tests environment state updates."""
    event = SAMPLE_EVENTS[0]
    proactive_agent.update_environment(event)
    
    assert len(proactive_agent.environment_state["events"]) == 1
    assert proactive_agent.environment_state["events"][0] == event

    # Test rolling window
    for _ in range(105):  # More than window size
        proactive_agent.update_environment({"test": "event"})
    
    assert len(proactive_agent.environment_state["events"]) == 100

def test_detect_user_needs(proactive_agent):
    """Tests the detection of user needs from events."""
    # Add sample events to environment
    for event in SAMPLE_EVENTS:
        proactive_agent.update_environment(event)
    
    detected_need = proactive_agent.detect_user_needs(SAMPLE_EVENTS)
    assert detected_need == "async/await template assistance"

def test_draft_proactive_task(proactive_agent):
    """Tests the creation of task proposals."""
    task_proposal = proactive_agent.draft_proactive_task(
        "async/await template assistance"
    )
    
    assert isinstance(task_proposal, dict)
    assert all(key in task_proposal 
              for key in ["task", "value_proposition", "execution_plan"])

@pytest.mark.asyncio
async def test_process_request_proactively(proactive_agent):
    """Tests the main proactive request processing workflow."""
    # Add sample events
    for event in SAMPLE_EVENTS:
        proactive_agent.update_environment(event)
    
    # Process request
    results = []
    async for result in proactive_agent.process_request_proactively(
        "Help me with Python programming"
    ):
        results.append(result)
    
    # Verify results contain both standard and proactive responses
    assert len(results) > 0
    
    # Check for proactive suggestion
    proactive_suggestions = [
        r for r in results 
        if r.get("type") == "proactive_suggestion"
    ]
    assert len(proactive_suggestions) > 0
    
    # Verify suggestion content
    suggestion = proactive_suggestions[0]
    assert "help" in suggestion["content"].lower()
    assert "would" in suggestion["content"].lower()

def test_process_proactive_feedback(proactive_agent):
    """Tests the feedback processing mechanism."""
    feedback = "Yes, that template was very helpful. Great timing!"
    proactive_agent.process_proactive_feedback(feedback, SAMPLE_TASK_PROPOSAL)
    
    # Verify learnings were stored
    assert "learnings" in proactive_agent.environment_state
    assert len(proactive_agent.environment_state["learnings"]) > 0

def test_error_handling(proactive_agent):
    """Tests error handling in proactive functions."""
    # Test with invalid events
    result = proactive_agent.detect_user_needs([])
    assert result is None
    
    # Test with malformed event
    proactive_agent.update_environment({"invalid": "event"})
    result = proactive_agent.detect_user_needs([{"invalid": "event"}])
    assert result is None

def test_threshold_behavior(proactive_agent):
    """Tests threshold-based decision making."""
    # Mock low confidence response
    with patch.object(
        proactive_agent.groq, 
        'generate', 
        return_value=json.dumps({
            "detected_need": "some need",
            "confidence": 0.5,  # Below threshold
            "reasoning": "test"
        })
    ):
        result = proactive_agent.detect_user_needs(SAMPLE_EVENTS)
        assert result is None

    # Mock high confidence response
    with patch.object(
        proactive_agent.groq, 
        'generate', 
        return_value=json.dumps({
            "detected_need": "some need",
            "confidence": 0.8,  # Above threshold
            "reasoning": "test"
        })
    ):
        result = proactive_agent.detect_user_needs(SAMPLE_EVENTS)
        assert result is not None

if __name__ == "__main__":
    pytest.main([__file__])