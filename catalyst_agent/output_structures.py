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
