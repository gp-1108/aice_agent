"""
LangGraph agent definition.
"""

from langgraph.graph import StateGraph, START, END
from catalyst_agent.state import AgentState
from catalyst_agent.tools import (
    parse_requirements,
	estimate_complexity,
	generate_tasks,
	create_acceptance_criteria,
	generate_prompt_for_copilot
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
		Dictionary with updated tasks
	"""
	parsed_reqs = state.get("parsed_requirements", {})
	complexities = state.get("estimated_complexities", [])
	tasks = generate_tasks.invoke({
		"features": parsed_reqs.features,
		"complexities": complexities
	})
	
	return {"tasks": tasks}

def create_acceptance_criteria_node(state: AgentState) -> dict:
	"""Node that creates acceptance criteria using the create_acceptance_criteria tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated acceptance_criteria
	"""
	parsed_reqs = state.get("parsed_requirements", {})
	tasks = state.get("tasks", [])
	acceptance_criteria = create_acceptance_criteria.invoke({
		"features": parsed_reqs.features,
		"tasks": tasks
	})
	
	return {"acceptance_criteria": acceptance_criteria}

def generate_copilot_prompts_node(state: AgentState) -> dict:
	"""Node that generates Copilot prompts using the generate_prompt_for_copilot tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated copilot_prompts
	"""
	parsed_reqs = state.get("parsed_requirements", {})
	tasks = state.get("tasks", [])
	acceptance_criteria = state.get("acceptance_criteria", [])
	copilot_prompts = generate_prompt_for_copilot.invoke({
		"features": parsed_reqs.features,
		"tasks": tasks,
		"acceptance_criteria": acceptance_criteria
	})
	
	return {"copilot_prompts": copilot_prompts}


def create_agent():
	"""Create and compile the LangGraph agent.
	
	Returns:
		Compiled LangGraph agent
	"""
	# Create the graph
	workflow = StateGraph(AgentState)
	
	# Add the nodes
	workflow.add_node("parse_requirements", parse_req_node)
	workflow.add_node("estimate_complexity", estimate_complexity_node)
	workflow.add_node("generate_tasks", generate_tasks_node)
	workflow.add_node("create_acceptance_criteria", create_acceptance_criteria_node)
	workflow.add_node("generate_copilot_prompts", generate_copilot_prompts_node)
	
	# Define the flow: START -> parse -> estimate -> tasks -> acceptance -> copilot -> END
	workflow.add_edge(START, "parse_requirements")
	workflow.add_edge("parse_requirements", "estimate_complexity")
	workflow.add_edge("estimate_complexity", "generate_tasks")
	workflow.add_edge("generate_tasks", "create_acceptance_criteria")
	workflow.add_edge("create_acceptance_criteria", "generate_copilot_prompts")
	workflow.add_edge("generate_copilot_prompts", END)
	
	# Compile and return
	return workflow.compile()
