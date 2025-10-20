"""
LangGraph agent definition.
"""

from langgraph.graph import StateGraph, START, END
from catalyst_agent.state import AgentState
from catalyst_agent.tools import parse_requirements


def parse_node(state: AgentState) -> dict:
	"""Node that parses requirements using the parse_requirements tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated parsed_data
	"""
	raw_text = state.get("raw_requirement", "")
	parsed_result = parse_requirements.invoke({"raw_text": raw_text})
	
	return {"parsed_data": parsed_result}


def create_agent():
	"""Create and compile the LangGraph agent.
	
	Returns:
		Compiled LangGraph agent
	"""
	# Create the graph
	workflow = StateGraph(AgentState)
	
	# Add the parse node
	workflow.add_node("parse", parse_node)
	
	# Define the flow: START -> parse -> END
	workflow.add_edge(START, "parse")
	workflow.add_edge("parse", END)
	
	# Compile and return
	return workflow.compile()
