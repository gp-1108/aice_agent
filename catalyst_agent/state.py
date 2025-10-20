"""
State definition for the agent workflow.
"""

from typing import TypedDict
from catalyst_agent.output_structures import (
	ParsedRequirements,
	EstimatedComplexity
)


class AgentState(TypedDict):
	"""State for the catalyst agent workflow."""
	raw_text: str
	parsed_requirements: ParsedRequirements | None
	estimated_complexities: list[EstimatedComplexity] | None
