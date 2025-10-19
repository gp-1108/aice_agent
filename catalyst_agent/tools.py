"""
Tools for the catalyst agent.
"""

from langchain_core.tools import tool


@tool
def parse_requirements(raw_text: str) -> dict:
    """Parse raw requirements text and extract key features.
    
    Args:
        raw_text: Raw requirement text to parse
        
    Returns:
        Dictionary containing parsed features
    """
    # Dummy implementation for testing
    return {"features": ["dummy feature"]}
