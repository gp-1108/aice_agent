"""
Tools for the catalyst agent.
"""

from langchain_core.tools import tool
from catalyst_agent.core import get_llm
from catalyst_agent.output_structures import (
	ParsedRequirements,
	EstimatedComplexity,
	Task,
	Feature,
	FeatureAcceptanceCriteria
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
def estimate_complexity(requirements: ParsedRequirements, raw_text: str) -> list[EstimatedComplexity]:
	"""Estimate the complexity of the project based on parsed requirements.
	
	Args:
		requirements: Parsed requirements object
		raw_text: Original raw requirement text
		
	Returns:
		List of complexity estimations per feature
	"""
	llm = get_llm().with_structured_output(EstimatedComplexity)
	complexities = []
	for feature in requirements.features:
		system_message = P.COMPLEXITY_ESTIMATION_SYSTEM_PROMPT
		user_message = P.COMPLEXITY_ESTIMATION_PROMPT.format(
			parsed_requirements=str(feature.model_dump_json(indent=2)),
			raw_text=raw_text
		)
		final_msg = P.GENERAL_SYSTEM_AND_USER_PROMPT.format(
			system_message=system_message,
			user_message=user_message
		)

		response = llm.invoke(final_msg)
		complexities.append(response)
	return complexities

@tool
def generate_tasks(features: list[Feature], complexities: list[EstimatedComplexity]) -> list[list[Task]]:
	"""Generate a list of tasks from parsed requirements.
	Break a feature into granular tasks with short descriptions and an initial
	guess at dependencies and priority
	
	Args:
		features: List of parsed features
		complexities: List of estimated complexities for each feature

	Returns:
		List of task lists (one list of tasks per feature)
	"""
	from catalyst_agent.output_structures import TaskNameList, TaskList
	
	tasks = []
	
	for feature, complexity in zip(features, complexities):
		# First LLM call: Generate task names only
		task_name_llm = get_llm().with_structured_output(TaskNameList)
		message = P.TASK_GENERATION_PROMPT.format(
			parsed_feature=str(feature.model_dump_json(indent=2)),
			estimated_complexity=str(complexity.model_dump_json(indent=2))
		)
		task_names_response = task_name_llm.invoke(message)
		
		# Second LLM call: Generate all detailed tasks at once
		task_detail_llm = get_llm().with_structured_output(TaskList)
		system_message = P.TASK_DETAILED_SYSTEM_PROMPT
		user_message = P.TASK_DETAILED_PROMPT.format(
			feature_context=str(feature.model_dump_json(indent=2)),
			all_task_titles="\n".join(f"- {title}" for title in task_names_response.tasks)
		)
		final_msg = P.GENERAL_SYSTEM_AND_USER_PROMPT.format(
			system_message=system_message,
			user_message=user_message
		)
		detailed_tasks_response = task_detail_llm.invoke(final_msg)
		
		tasks.append(detailed_tasks_response.tasks)
	
	return tasks

@tool
def create_acceptance_criteria(features: list[Feature], tasks: list[list[Task]]) -> list[FeatureAcceptanceCriteria]:
	"""Create acceptance criteria and test descriptions for all tasks.
	
	Generate clear, testable acceptance criteria using Given/When/Then format
	along with comprehensive unit and integration test descriptions for each task.
	
	Args:
		features: List of parsed features
		tasks: List of task lists (one list per feature)
		
	Returns:
		List of acceptance criteria for each feature's tasks
	"""
	llm = get_llm().with_structured_output(FeatureAcceptanceCriteria)
	acceptance_criteria_list = []
	
	for feature, feature_tasks in zip(features, tasks):
		# Format tasks for the prompt
		tasks_formatted = "\n".join([
			f"- {task.title}: {task.description}"
			for task in feature_tasks
		])
		
		system_message = P.ACCEPTANCE_CRITERIA_SYSTEM_PROMPT
		user_message = P.ACCEPTANCE_CRITERIA_PROMPT.format(
			feature_context=str(feature.model_dump_json(indent=2)),
			tasks_list=tasks_formatted
		)
		final_msg = P.GENERAL_SYSTEM_AND_USER_PROMPT.format(
			system_message=system_message,
			user_message=user_message
		)
		
		response = llm.invoke(final_msg)
		acceptance_criteria_list.append(response)
	
	return acceptance_criteria_list

