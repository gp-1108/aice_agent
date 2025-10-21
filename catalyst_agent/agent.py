"""
LangGraph agent definition.
"""

import json
from langgraph.graph import StateGraph, START, END
from catalyst_agent.state import AgentState
from catalyst_agent.output_structures import ProjectPhase
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
	print("[Agent Step 1/5] Parsing requirements...")
	raw_text = state.get("raw_text", "")
	parsed_result = parse_requirements.invoke({"raw_text": raw_text})
	print("  ✓ Requirements parsed")
	
	return {"parsed_requirements": parsed_result}

def estimate_complexity_node(state: AgentState) -> dict:
	"""Node that estimates complexity using the estimate_complexity tool per each feature extracted.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated final_plan
	"""
	print("[Agent Step 2/5] Estimating complexity for each feature...")
	parsed_reqs = state.get("parsed_requirements")
	raw_text = state.get("raw_text", "")
	complexity = estimate_complexity.invoke({
		"requirements": parsed_reqs,
		"raw_text": raw_text
	})
	print("  ✓ Complexity estimated")
	
	return {"estimated_complexities": complexity}

def generate_tasks_node(state: AgentState) -> dict:
	"""Node that generates tasks using the generate_tasks tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated tasks
	"""
	print("[Agent Step 3/5] Generating tasks for each feature...")
	parsed_reqs = state.get("parsed_requirements")
	complexities = state.get("estimated_complexities", [])
	tasks = generate_tasks.invoke({
		"features": parsed_reqs.features if parsed_reqs else [],
		"complexities": complexities
	})
	print("  ✓ Tasks generated")
	
	return {"tasks": tasks}

def create_acceptance_criteria_node(state: AgentState) -> dict:
	"""Node that creates acceptance criteria using the create_acceptance_criteria tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated acceptance_criteria
	"""
	print("[Agent Step 4/5] Creating acceptance criteria for tasks...")
	parsed_reqs = state.get("parsed_requirements")
	tasks = state.get("tasks", [])
	acceptance_criteria = create_acceptance_criteria.invoke({
		"features": parsed_reqs.features if parsed_reqs else [],
		"tasks": tasks
	})
	print("  ✓ Acceptance criteria created")
	
	return {"acceptance_criteria": acceptance_criteria}

def generate_copilot_prompts_node(state: AgentState) -> dict:
	"""Node that generates Copilot prompts using the generate_prompt_for_copilot tool.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with updated copilot_prompts
	"""
	print("[Agent Step 5/5] Generating Copilot prompts...")
	parsed_reqs = state.get("parsed_requirements")
	tasks = state.get("tasks", [])
	acceptance_criteria = state.get("acceptance_criteria", [])
	copilot_prompts = generate_prompt_for_copilot.invoke({
		"features": parsed_reqs.features if parsed_reqs else [],
		"tasks": tasks,
		"acceptance_criteria": acceptance_criteria
	})
	print("  ✓ Copilot prompts generated")
	
	return {"copilot_prompts": copilot_prompts}

def generate_final_json_node(state: AgentState) -> dict:
	"""Node that generates the final JSON output.
	
	Args:
		state: Current agent state
		
	Returns:
		Dictionary with final JSON output
	"""
	print("[Agent] Generating final JSON output...")
	final_dict = {}

	# Metadata
	final_dict["metadata"] = {}
	metadata = final_dict["metadata"]
	metadata["original_text"] = state.get("raw_text", "")
	
	parsed_reqs = state.get("parsed_requirements")
	metadata["num_features"] = len(parsed_reqs.features) if parsed_reqs else 0
	metadata["num_tasks"] = 0
	if state.get("tasks"):
		metadata["num_tasks"] = sum(len(task_group) for task_group in state.get("tasks", []))
	
	# Raw values
	metadata["raw_values"] = {}
	raw_values = metadata["raw_values"]
	raw_values["parsed_requirements"] = state.get("parsed_requirements", {})
	raw_values["estimated_complexities"] = state.get("estimated_complexities", [])
	raw_values["tasks"] = state.get("tasks", [])
	raw_values["acceptance_criteria"] = state.get("acceptance_criteria", [])
	raw_values["copilot_prompts"] = state.get("copilot_prompts", [])

	# Gathering all info of tasks into a single dictionary
	tasks_dict = {}
	for task_group in state.get("tasks", []):
		for task in task_group:
			tasks_dict[task.title] = {
				"description": task.description,
				"priority": task.priority,
				"dependencies": task.dependencies,
				"phase": task.phase,
				"acceptance_criteria": None,
				"unit_tests": [],
				"integration_tests": [],
				"copilot_prompt": None
			}
	# Adding acceptance criteria and tests
	for feature_criteria in state.get("acceptance_criteria", []):
		for task_criteria in feature_criteria.tasks_criteria:
			if task_criteria.task_title in tasks_dict:
				tasks_dict[task_criteria.task_title]["acceptance_criteria"] = task_criteria.acceptance_criteria
				tasks_dict[task_criteria.task_title]["unit_tests"] = task_criteria.unit_tests
				tasks_dict[task_criteria.task_title]["integration_tests"] = task_criteria.integration_tests
	# Adding Copilot prompts
	for feature_prompts in state.get("copilot_prompts", []):
		for task_prompt in feature_prompts.task_prompts:
			if task_prompt.task_title in tasks_dict:
				tasks_dict[task_prompt.task_title]["copilot_prompt"] = task_prompt.prompt

	# Now adding the tasks into phases
	final_dict["development_plan"] = {}
	for phase in ProjectPhase:
		final_dict["development_plan"][phase.value] = {}
	# Populate tasks into their respective phases
	for task_title, details in tasks_dict.items():
		phase = details["phase"]
		if phase in final_dict["development_plan"]:
			final_dict["development_plan"][phase][task_title] = {
				"description": details["description"],
				"priority": details["priority"],
				"dependencies": details["dependencies"],
				"acceptance_criteria": details["acceptance_criteria"],
				"unit_tests": details["unit_tests"],
				"integration_tests": details["integration_tests"],
				"copilot_prompt": details["copilot_prompt"]
			}
	json_str = json.dumps(final_dict, indent=4, default=str)
	print("  ✓ Final JSON generated")
	print("[Agent] ✓ All steps completed successfully!")
	return {"final_json": json_str}

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
	workflow.add_node("generate_final_json", generate_final_json_node)
	
	# Define the flow: START -> parse -> estimate -> tasks -> acceptance -> copilot -> END
	workflow.add_edge(START, "parse_requirements")
	workflow.add_edge("parse_requirements", "estimate_complexity")
	workflow.add_edge("estimate_complexity", "generate_tasks")
	workflow.add_edge("generate_tasks", "create_acceptance_criteria")
	workflow.add_edge("create_acceptance_criteria", "generate_copilot_prompts")
	workflow.add_edge("generate_copilot_prompts", "generate_final_json")
	workflow.add_edge("generate_final_json", END)

	# Compile and return
	return workflow.compile()
