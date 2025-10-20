"""
MLflow Model Logging Script for Catalyst Agent (Models From Code Approach)

This script demonstrates how to:
1. Log a LangGraph agent as an MLflow LangChain flavor model using "models from code"
2. Reload the model from MLflow
3. Use the reloaded model for inference in a clean process

The "models from code" approach:
- Avoids serialization/pickling issues (especially with RunnableLambda, custom functions)
- Ensures compatibility across Python versions
- Makes the model definition human-readable
- Works seamlessly with LangGraph's CompiledStateGraph

Reference: https://mlflow.org/docs/latest/llms/langchain/index.html
"""

import mlflow
from mlflow.models import infer_signature
import sys
import os
from pathlib import Path

# =====================================
# CONFIGURATION
# =====================================
EXPERIMENT_NAME = "catalyst-agent-model-logging"
MODEL_NAME = "catalyst-agent"
AGENT_MODEL_PATH = "agent_model.py"  # Path to file defining the agent


def log_agent_to_mlflow(verbose=True):
	"""
	Log the Catalyst Agent to MLflow using the "models from code" approach.
	
	Args:
		verbose: If True, print detailed progress information
		
	Returns:
		tuple: (run_id, model_uri) - The MLflow run ID and model URI
	"""
	# Get the directory where this script is located
	script_dir = Path(__file__).parent
	agent_model_path = script_dir / AGENT_MODEL_PATH
	
	if verbose:
		print("="*70)
		print("MLflow Model Logging for Catalyst Agent")
		print("Using 'Models From Code' Approach")
		print("="*70)

	if verbose:
		print("="*70)
		print("MLflow Model Logging for Catalyst Agent")
		print("Using 'Models From Code' Approach")
		print("="*70)

	# =====================================
	# 1. ENABLE MLFLOW AUTOLOGGING
	# =====================================
	if verbose:
		print("\n[Step 1] Enabling MLflow autologging for LangChain...")
	mlflow.langchain.autolog(
		log_traces=True,  # Enable trace logging
		silent=not verbose       # Show MLflow logs if verbose
	)
	if verbose:
		print("✓ Autologging enabled")

	# =====================================
	# 2. SET EXPERIMENT
	# =====================================
	if verbose:
		print(f"\n[Step 2] Setting MLflow experiment: {EXPERIMENT_NAME}")
	mlflow.set_experiment(EXPERIMENT_NAME)
	experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
	if verbose:
		print(f"✓ Experiment ID: {experiment.experiment_id}")

	# =====================================
	# 3. VERIFY AGENT MODEL FILE
	# =====================================
	if verbose:
		print(f"\n[Step 3] Verifying agent model file: {agent_model_path}")
	if not agent_model_path.exists():
		error_msg = f"❌ Error: Agent model file '{agent_model_path}' not found!"
		if verbose:
			print(error_msg)
		raise FileNotFoundError(error_msg)
	if verbose:
		print(f"✓ Agent model file found")

	# =====================================
	# 4. LOAD AND TEST AGENT
	# =====================================
	if verbose:
		print("\n[Step 4] Loading agent from model file for testing...")
	
	# Import the agent to test it before logging
	# Add the mlflow_logging directory to the path temporarily
	import importlib.util
	spec = importlib.util.spec_from_file_location("agent_model", agent_model_path)
	agent_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(agent_module)
	agent = agent_module.agent
	
	if verbose:
		print(f"✓ Agent loaded: {type(agent).__name__}")

	# =====================================
	# 5. PREPARE SAMPLE INPUT/OUTPUT FOR SIGNATURE
	# =====================================
	if verbose:
		print("\n[Step 5] Preparing model signature...")

	# Sample input for signature inference
	sample_input = {
		"raw_requirement": "Build a web application with user authentication",
		"parsed_data": None,
		"final_plan": None
	}

	# Generate sample output
	if verbose:
		print("  Running test inference...")
	sample_output = agent.invoke(sample_input)
	if verbose:
		print(f"  Test output: {sample_output}")

	# Infer signature from input/output
	signature = infer_signature(sample_input, sample_output)
	if verbose:
		print("✓ Model signature inferred:")
		print(f"  - Input schema: {signature.inputs}")
		print(f"  - Output schema: {signature.outputs}")

	# =====================================
	# 6. LOG THE MODEL TO MLFLOW (FROM CODE)
	# =====================================
	if verbose:
		print("\n[Step 6] Logging model to MLflow from code...")

	with mlflow.start_run(run_name="log-catalyst-agent-from-code") as run:
		run_id = run.info.run_id
		if verbose:
			print(f"✓ Started run: {run_id}")
		
		# Log the agent as a LangChain model FROM CODE
		# The key difference: we pass the path to the Python file, not the object
		model_info = mlflow.langchain.log_model(
			lc_model=str(agent_model_path),  # Pass file path, not the object!
			name=MODEL_NAME,
			signature=signature,
			input_example=sample_input,
			pip_requirements=[
				"langchain==0.3.27",
				"langchain-core==0.3.79",
				"langchain-openai==0.3.27",
				"langgraph==0.3.27",
				"mlflow>=3.5.0",
				"pydantic>=2.10.4",
			],
		)
		
		if verbose:
			print(f"✓ Model logged successfully!")
			print(f"  - Model URI: {model_info.model_uri}")
			print(f"  - Run ID: {run_id}")
			print(f"  - Artifact Path: {model_info.artifact_path}")

	# =====================================
	# 7. DOCUMENT LOGGED OBJECT (if verbose)
	# =====================================
	if verbose:
		print("\n" + "="*70)
		print("LOGGED OBJECT DOCUMENTATION")
		print("="*70)
		print(f"""
Object Type: langgraph.graph.state.CompiledStateGraph
Logging Method: Models From Code (MLflow 2.12.2+)
Flavor: mlflow.langchain

The logged object is a LangGraph CompiledStateGraph defined in code.

WHY "MODELS FROM CODE" APPROACH:
✓ Avoids serialization/pickling complications
✓ No issues with unpicklable objects (lambdas, file handles, etc.)
✓ Cross-Python-version compatibility guaranteed
✓ Human-readable model definition
✓ Easy to review and audit the model code

THE AGENT WORKFLOW:

1. Accepts input as AgentState dictionary:
   - raw_requirement (str): Raw requirement text to parse
   - parsed_data (dict | None): Placeholder for parsed data
   - final_plan (dict | None): Placeholder for final plan

2. Executes workflow: START -> parse -> END
   - The parse node calls the parse_requirements tool

3. Returns AgentState dictionary with:
   - raw_requirement: Original input
   - parsed_data: Parsed requirements from tool
   - final_plan: Final plan (currently None)

MODEL STORAGE:
- The agent code is stored in: {AGENT_MODEL_PATH}
- This code is packaged with the MLflow model artifact
- Anyone can review the exact model definition

Model URI: {model_info.model_uri}
Run ID: {run_id}
""")

	# =====================================
	# 8. RELOAD AND TEST THE MODEL (if verbose)
	# =====================================
	if verbose:
		print("="*70)
		print("MODEL RELOAD AND INFERENCE TEST")
		print("="*70)

		print(f"\n[Step 7] Reloading model from MLflow...")
	
	loaded_model = mlflow.langchain.load_model(model_info.model_uri)
	
	if verbose:
		print(f"✓ Model reloaded successfully!")
		print(f"  - Type: {type(loaded_model).__name__}")

		# Test the reloaded model
		print(f"\n[Step 8] Testing reloaded model with inference...")
		test_input = {
			"raw_requirement": "Create a mobile app with push notifications and offline mode",
			"parsed_data": None,
			"final_plan": None
		}

		print(f"\nInput:")
		print(f"  raw_requirement: {test_input['raw_requirement']}")

		result = loaded_model.invoke(test_input)

		print(f"\nOutput:")
		print(f"  raw_requirement: {result['raw_requirement']}")
		print(f"  parsed_data: {result['parsed_data']}")
		print(f"  final_plan: {result['final_plan']}")

		print("\n✓ Inference successful!")

	# =====================================
	# 9. SUMMARY AND INSTRUCTIONS (if verbose)
	# =====================================
	if verbose:
		print("\n" + "="*70)
		print("SUMMARY AND NEXT STEPS")
		print("="*70)
		print(f"""
✅ Model successfully logged and tested using "Models From Code"!

WHAT WAS LOGGED:
- Object: LangGraph CompiledStateGraph
- Method: Models From Code (no pickling!)
- Source: {AGENT_MODEL_PATH}
- Flavor: mlflow.langchain
- Location: {model_info.model_uri}
- Run ID: {run_id}

VERIFICATION COMPLETED:
✓ Model logged to MLflow tracking server
✓ Model reloaded in the same process
✓ Model successfully used for inference
✓ Input/output signatures captured
✓ Code is human-readable in MLflow UI

ADVANTAGES OF THIS APPROACH:
✓ No serialization issues with complex objects
✓ Works across different Python versions
✓ Model definition is transparent and auditable
✓ Easy to review what the model does
✓ Perfect for LangGraph agents with custom nodes

TO RELOAD IN A CLEAN PROCESS:
-----------------------------------------------------------------------
import mlflow

# Load the model
model_uri = "runs:/{run_id}/{MODEL_NAME}"
loaded_agent = mlflow.langchain.load_model(model_uri)

# Use for inference
result = loaded_agent.invoke({{
	"raw_requirement": "Your requirement text here",
	"parsed_data": None,
	"final_plan": None
}})

print(result)
-----------------------------------------------------------------------

TO VIEW IN MLFLOW UI:
-----------------------------------------------------------------------
1. Start MLflow UI: mlflow ui
2. Open: http://localhost:5000
3. Navigate to experiment: {EXPERIMENT_NAME}
4. Click on run: {run_id}
5. View the model code in the Artifacts section
6. The actual Python code will be visible and readable!
-----------------------------------------------------------------------

TO VERIFY IN A CLEAN PROCESS:
-----------------------------------------------------------------------
python -m mlflow_logging.verify_model_reload {run_id}
-----------------------------------------------------------------------

TO DEPLOY THE MODEL:
The logged model can be deployed using:
- MLflow Model Serving (local)
- Cloud deployment (AWS SageMaker, Azure ML, etc.)
- Kubernetes with MLflow
- Custom serving infrastructure

For more information, see:
https://mlflow.org/docs/latest/models.html#deployment
""")

		print("="*70)
		print("Script completed successfully! ✨")
		print("="*70)
	
	return run_id, model_info.model_uri


if __name__ == "__main__":
	# Run as standalone script
	try:
		run_id, model_uri = log_agent_to_mlflow(verbose=True)
		sys.exit(0)
	except Exception as e:
		print(f"\n❌ Error: {e}")
		sys.exit(1)
