"""
State definition for the agent workflow.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """State for the catalyst agent workflow."""
    
    raw_requirement: str
    parsed_data: dict | None
    final_plan: dict | None
