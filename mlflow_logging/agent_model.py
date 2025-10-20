"""
Agent Model Definition for MLflow Logging

This file defines the LangGraph agent that will be logged to MLflow using
the "models from code" approach. This avoids serialization issues and makes
the model definition human-readable.

The agent is automatically set as the MLflow model using mlflow.models.set_model()
"""

from langgraph.graph import StateGraph, START, END
from catalyst_agent.state import AgentState
from catalyst_agent.tools import parse_requirements
import mlflow


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


# Create the agent instance
agent = create_agent()

# Set this agent as the model to be logged by MLflow
# This is the key step for "models from code" logging
mlflow.models.set_model(agent)
