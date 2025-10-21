"""
Agent Model Definition for MLflow Logging

This file imports the actual Catalyst Agent from the catalyst_agent module
and sets it as the MLflow model using mlflow.models.set_model().

This ensures that the logged model is exactly the same as the one used in production.
"""

from catalyst_agent.agent import create_agent
import mlflow


# Create the agent instance using the actual agent factory
agent = create_agent()

# Set this agent as the model to be logged by MLflow
# This is the key step for "models from code" logging
mlflow.models.set_model(agent)
