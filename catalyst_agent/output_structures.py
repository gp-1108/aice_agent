from pydantic import BaseModel
from enum import Enum

class Feature(BaseModel):
	name: str
	description: str

class ParsedRequirements(BaseModel):
	features: list[Feature]
	constraints: list[str]
	stakeholders: list[str]
	success_criteria: list[str]

class Difficulty(str, Enum):
	EASY = "easy"
	MEDIUM = "medium"
	HARD = "hard"
	VERY_HARD = "very_hard"

class EstimatedComplexity(BaseModel):
	"""Estimated complexity of the project."""
	difficulty: Difficulty
	estimated_days: int
	risks: list[str]

class TaskNameList(BaseModel):
	tasks: list[str]

class TaskPriority(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"

class Task(BaseModel):
	title: str
	description: str
	priority: TaskPriority
	dependencies: list[str]

class TaskList(BaseModel):
	"""A list of tasks for a feature."""
	tasks: list[Task]

class TestType(str, Enum):
	UNIT = "unit"
	INTEGRATION = "integration"

class TestDescription(BaseModel):
	"""Description of a test case."""
	test_name: str
	test_type: TestType
	description: str

class AcceptanceCriterion(BaseModel):
	"""A single acceptance criterion in Given/When/Then format."""
	given: str
	when: str
	then: str

class TaskAcceptanceCriteria(BaseModel):
	"""Acceptance criteria and tests for a single task."""
	task_title: str
	acceptance_criteria: list[AcceptanceCriterion]
	unit_tests: list[TestDescription]
	integration_tests: list[TestDescription]

class FeatureAcceptanceCriteria(BaseModel):
	"""Acceptance criteria for all tasks in a feature."""
	feature_name: str
	tasks_criteria: list[TaskAcceptanceCriteria]
