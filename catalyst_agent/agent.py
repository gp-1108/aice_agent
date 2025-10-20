"""
LangGraph agent definition.
"""

from langgraph.graph import StateGraph, START, END
from catalyst_agent.state import AgentState
from catalyst_agent.tools import (
    parse_requirements,
	estimate_complexity,
	generate_tasks
)

def parse_req_node(state: AgentState) -> dict:
	"""Node that parses requirements using the parse_requirements tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated parsed_data
	"""
	raw_text = state.get("raw_requirement", "")
	parsed_result = parse_requirements.invoke({"raw_text": raw_text})
	
	return {"parsed_requirements": parsed_result}

def estimate_complexity_node(state: AgentState) -> dict:
	"""Node that estimates complexity using the estimate_complexity tool per each feature extracted.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated final_plan
	"""
	parsed_reqs = state.get("parsed_requirements", {})
	raw_text = state.get("raw_text", "")
	complexity = estimate_complexity.invoke({
		"requirements": parsed_reqs,
		"raw_text": raw_text
	})
	
	return {"estimated_complexities": complexity}

def generate_tasks_node(state: AgentState) -> dict:
	"""Node that generates tasks using the generate_tasks tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated final_plan
	"""
	parsed_reqs = state.get("parsed_requirements", {})
	complexities = state.get("estimated_complexities", [])
	tasks = generate_tasks.invoke({
		"features": parsed_reqs.features,
		"complexities": complexities
	})
	
	return {"tasks": tasks}


def create_agent():
	"""Create and compile the LangGraph agent.
	
	Returns:
		Compiled LangGraph agent
	"""
	# Create the graph
	workflow = StateGraph(AgentState)
	
	# Add the parse node
	workflow.add_node("parse_requirements", parse_req_node)
	workflow.add_node("estimate_complexity", estimate_complexity_node)
	workflow.add_node("generate_tasks", generate_tasks_node)
	
	# Define the flow: START -> parse -> END
	workflow.add_edge(START, "parse_requirements")
	workflow.add_edge("parse_requirements", "estimate_complexity")
	workflow.add_edge("estimate_complexity", "generate_tasks")
	workflow.add_edge("generate_tasks", END)
	
	# Compile and return
	return workflow.compile()
