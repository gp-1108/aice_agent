"""
Tools for the catalyst agent.
"""

from langchain_core.tools import tool
from catalyst_agent.core import get_llm
from catalyst_agent.output_structures import (
    ParsedRequirements
)
from catalyst_agent import prompts as P


@tool
def parse_requirements(raw_text: str) -> ParsedRequirements:
    """Parse raw requirements text and extract key features.
    
    Extract features, constraints, stakeholders, and success criteria from
    the raw text into a simple structured object.
    Args:
        raw_text: Raw requirement text to parse
        
    Returns:
        Dictionary containing parsed features
    """
    llm = get_llm().with_structured_output(ParsedRequirements)
    system_message = P.REQUIREMENTS_SYSTEM_PROMPT
    user_message = P.REQUIREMENTS_PROMPT.format(raw_input=raw_text)
    final_msg = P.GENERAL_SYSTEM_AND_USER_PROMPT.format(
        system_message=system_message,
        user_message=user_message
    )

    response = llm.invoke(final_msg)
    return response

@tool
def estimate_complexity_tool(requirements: ParsedRequirements, raw_text: str) -> str:
    """Estimate the complexity of the project based on parsed requirements.
    
    Args:
        requirements: Parsed requirements object
        raw_text: Original raw requirement text
        
    Returns:
        str: Estimated complexity level (e.g., "Low", "Medium", "High")
    """
    llm = get_llm()
    system_message = "You are an expert project manager. Based on the provided project requirements, estimate the overall complexity of the project as Low, Medium, or High."
    user_message = f"The project has the following features: {', '.join([f.name for f in requirements.features])}. Consider also the constraints: {', '.join(requirements.constraints)}. Based on this information, provide an estimated complexity level."
    final_msg = P.GENERAL_SYSTEM_AND_USER_PROMPT.format(
        system_message=system_message,
        user_message=user_message
    )

    response = llm.invoke(final_msg)
    return response.strip()