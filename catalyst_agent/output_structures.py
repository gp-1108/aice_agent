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
