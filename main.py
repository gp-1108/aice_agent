"""
Catalyst Agent Main - Interactive Demo

This demonstrates the LangGraph agent with MLflow integration options:
1. Run the agent normally with automatic tracing
2. Log the agent as an MLflow model
3. View/load previously logged models
"""

import sys
import mlflow
import json
from datetime import datetime
from pathlib import Path
from catalyst_agent.agent import create_agent


def save_result_to_json(result: dict, run_id: str) -> str:
	"""Save the agent result to a JSON file.
	
	Args:
		result: The agent's final state dictionary
		run_id: MLflow run ID for filename
		
	Returns:
		Path to the saved JSON file
	"""
	# Create output directory if it doesn't exist
	output_dir = Path("agent_outputs")
	output_dir.mkdir(exist_ok=True)
	
	# Create filename with timestamp
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"agent_result_{timestamp}_{run_id[:8]}.json"
	filepath = output_dir / filename
	
	# Convert result to JSON-serializable format
	json_result = {
		"run_id": run_id,
		"timestamp": timestamp,
		"raw_text": result.get("raw_text", ""),
		"parsed_requirements": result.get("parsed_requirements").model_dump() if result.get("parsed_requirements") else None,
		"estimated_complexities": [c.model_dump() for c in result.get("estimated_complexities", [])],
		"tasks": [[task.model_dump() for task in task_list] for task_list in result.get("tasks", [])],
		"acceptance_criteria": [ac.model_dump() for ac in result.get("acceptance_criteria", [])],
		"copilot_prompts": [cp.model_dump() for cp in result.get("copilot_prompts", [])]
	}
	
	# Save to file
	with open(filepath, 'w', encoding='utf-8') as f:
		json.dump(json_result, f, indent=2, ensure_ascii=False)
	
	print(f"✓ Results saved to: {filepath}")
	return str(filepath)


def print_menu():
	"""Display the main menu."""
	print("\n" + "="*70)
	print("CATALYST AGENT - Main Menu")
	print("="*70)
	print("\n1. Run Agent (with MLflow tracing)")
	print("2. Log Agent to MLflow")
	print("3. View/Load Logged Models")
	print("4. Exit")
	print("\n" + "="*70)


def run_agent():
	"""Run the agent with a sample requirement (Option 1)."""
	print("\n" + "="*70)
	print("RUNNING AGENT WITH MLFLOW TRACING")
	print("="*70)
	
	# Enable MLflow autologging for tracing
	print("\n[Step 1] Enabling MLflow autologging for LangChain...")
	mlflow.langchain.autolog(
		log_traces=True,  # Enable trace logging
		silent=False      # Show MLflow logs
	)
	print("✓ Autologging enabled")
	
	# Set experiment
	experiment_name = "catalyst-agent-demo"
	print(f"\n[Step 2] Setting MLflow experiment: {experiment_name}")
	mlflow.set_experiment(experiment_name)
	experiment = mlflow.get_experiment_by_name(experiment_name)
	print(f"✓ Experiment ID: {experiment.experiment_id}")
	
	# Create the agent
	print("\n[Step 3] Creating LangGraph agent...")
	agent = create_agent()
	print(f"✓ Agent created: {type(agent).__name__}")
	
	# Get requirement from user
	print("\n[Step 4] Getting user input...")
	print("Enter your requirement (or press Enter for demo):")
	user_input = input("> ").strip()
	
	if not user_input:
		user_input = "We need an internal dashboard for monitoring employee performance across departments. The dashboard should pull data from our HR database and show KPIs like productivity, completed projects, and average feedback ratings. Managers should be able to filter results by team, department, and time period. It must update daily and display charts and tables. The dashboard should be web-based and integrated with our company's SSO for authentication. Ideally, it should be ready within six weeks."
		print(f"Using demo requirement")
	
	# Prepare input data
	input_data = {
		"raw_text": user_input,
		"parsed_requirements": None,
		"estimated_complexities": None,
		"tasks": None,
		"acceptance_criteria": None,
		"copilot_prompts": None
	}
	
	print("\n" + "="*70)
	print("[Step 5] Running agent with MLflow tracking...")
	print("="*70)
	
	# Run the agent within an MLflow run context
	with mlflow.start_run(run_name="catalyst-agent-execution") as run:
		run_id = run.info.run_id
		print(f"✓ Started MLflow run: {run_id}")
		
		print("\nProcessing requirement...")
		result = agent.invoke(input_data)
		
		print("\n" + "="*70)
		print("AGENT FINAL STATE:")
		print("="*70)
		# Print the final state summary
		print(f"raw_text: {result.get('raw_text', '')[:100]}...")
		print(f"parsed_requirements: {len(result.get('parsed_requirements', {}).features if result.get('parsed_requirements') else [])} features")
		print(f"estimated_complexities: {len(result.get('estimated_complexities', []))} complexities")
		print(f"tasks: {sum(len(task_list) for task_list in result.get('tasks', []))} total tasks")
		print(f"acceptance_criteria: {len(result.get('acceptance_criteria', []))} feature criteria")
		print(f"copilot_prompts: {sum(len(cp.task_prompts) for cp in result.get('copilot_prompts', []))} prompts generated")
		
		# Save to JSON
		print("\n[Step 6] Saving results to JSON...")
		save_result_to_json(result, run_id)
		
		print(f"\n✓ Agent execution completed!")
		print(f"✓ MLflow Run ID: {run_id}")
		print(f"\nTo view this run in MLflow UI:")
		print(f"  1. Start MLflow UI: mlflow ui")
		print(f"  2. Navigate to: http://localhost:5000")
		print(f"  3. Look for experiment: catalyst-agent-demo")
		print(f"  4. Find run: {run_id}")
	
	return result


def log_agent():
	"""Log the agent to MLflow (Option 2)."""
	print("\n" + "="*70)
	print("LOGGING AGENT TO MLFLOW")
	print("="*70)
	
	try:
		from mlflow_logging import log_agent_to_mlflow
		
		print("\nThis will log the agent using the 'Models From Code' approach...")
		print("Continue? (y/n): ", end="")
		confirm = input().strip().lower()
		
		if confirm != 'y':
			print("Cancelled.")
			return
		
		print("\nLogging agent...")
		run_id, model_uri = log_agent_to_mlflow(verbose=True)
		
		print(f"\n{'='*70}")
		print("SUCCESS!")
		print(f"{'='*70}")
		print(f"Run ID: {run_id}")
		print(f"Model URI: {model_uri}")
		print(f"\nTo verify: python -m mlflow_logging.verify_model_reload {run_id}")
		
	except ImportError:
		print("\n❌ Error: mlflow_logging module not found")
		print("Make sure mlflow_logging/ directory exists with required files")
	except Exception as e:
		print(f"\n❌ Error logging agent: {e}")


def view_models():
	"""View and optionally load logged models (Option 3)."""
	print("\n" + "="*70)
	print("VIEW/LOAD LOGGED MODELS")
	print("="*70)
	
	try:
		# Get all experiments
		experiments = mlflow.search_experiments()
		
		# Find relevant experiment
		target_exp = None
		for exp in experiments:
			if "catalyst-agent" in exp.name:
				target_exp = exp
				break
		
		if not target_exp:
			print("\n⚠️  No catalyst-agent experiments found")
			print("Run option 2 to log the agent first")
			return
		
		print(f"\nExperiment: {target_exp.name}")
		print(f"Experiment ID: {target_exp.experiment_id}")
		
		# Get runs from this experiment
		runs = mlflow.search_runs(
			experiment_ids=[target_exp.experiment_id],
			order_by=["start_time DESC"],
			max_results=10
		)
		
		if runs.empty:
			print("\n⚠️  No runs found in this experiment")
			return
		
		print(f"\nFound {len(runs)} recent run(s):\n")
		
		for idx, run in runs.iterrows():
			run_id = run['run_id']
			start_time = run['start_time']
			status = run['status']
			print(f"{idx + 1}. Run ID: {run_id}")
			print(f"   Status: {status}")
			print(f"   Started: {start_time}")
			print()
		
		print("Options:")
		print("1. Load a specific model by run ID")
		print("2. Open MLflow UI")
		print("3. Back to main menu")
		
		choice = input("\nSelect (1-3): ").strip()
		
		if choice == "1":
			run_id = input("Enter run ID: ").strip()
			load_and_test_model(run_id)
		elif choice == "2":
			print("\nTo open MLflow UI, run in a terminal:")
			print("  mlflow ui")
			print("\nThen navigate to: http://localhost:5000")
		
	except Exception as e:
		print(f"\n❌ Error: {e}")


def load_and_test_model(run_id):
	"""Load and test a specific model."""
	print(f"\n{'='*70}")
	print(f"LOADING MODEL: {run_id}")
	print(f"{'='*70}")
	
	try:
		model_uri = f"runs:/{run_id}/catalyst-agent"
		print(f"\nLoading from: {model_uri}")
		
		agent = mlflow.langchain.load_model(model_uri)
		print(f"✓ Model loaded: {type(agent).__name__}")
		
		print("\nTest the model? (y/n): ", end="")
		if input().strip().lower() == 'y':
			test_input = input("\nEnter requirement (or press Enter for demo): ").strip()
			if not test_input:
				test_input = "Create a REST API with authentication"
			
			result = agent.invoke({
				"raw_requirement": test_input,
				"parsed_data": None,
				"final_plan": None
			})
			
			print("\n" + "-"*70)
			print("RESULT:")
			print("-"*70)
			print(f"Input: {result['raw_requirement']}")
			print(f"Parsed: {result['parsed_data']}")
			print(f"Plan: {result['final_plan']}")
			print("\n✓ Inference successful!")
			
	except Exception as e:
		print(f"\n❌ Error: {e}")


def main():
	"""Main entry point with menu system."""
	print("\n" + "="*70)
	print("CATALYST AGENT - LangGraph with MLflow")
	print("="*70)
	
	while True:
		print_menu()
		choice = input("Select an option (1-4): ").strip()
		
		if choice == "1":
			run_agent()
		elif choice == "2":
			log_agent()
		elif choice == "3":
			view_models()
		elif choice == "4":
			print("\nExiting... Goodbye! 👋")
			sys.exit(0)
		else:
			print("\n❌ Invalid choice. Please select 1-4.")
		
		input("\nPress Enter to continue...")


if __name__ == "__main__":
	main()
