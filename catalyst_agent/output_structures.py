from pydantic import BaseModel

class Feature(BaseModel):
	name: str
	description: str

class ParsedRequirements(BaseModel):
	features: list[Feature]
	constraints: list[str]
	stakeholders: list[str]
	success_criteria: list[str]
