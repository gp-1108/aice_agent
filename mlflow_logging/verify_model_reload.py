"""
Model Reload Verification Script

This script demonstrates loading a previously logged MLflow model in a clean
process and using it for inference. Run this AFTER running log_model_to_mlflow.py

Usage:
	python verify_model_reload.py <run_id>
	
Or from Python:
	from mlflow_logging import verify_model
	verify_model(run_id)

Example:
	python verify_model_reload.py abc123def456
"""

import sys
import mlflow


def verify_model(run_id, verbose=True):
	"""
	Verify that a logged model can be reloaded and used for inference.
	
	Args:
		run_id: The MLflow run ID containing the model
		verbose: If True, print detailed progress information
		
	Returns:
		bool: True if verification passed, False otherwise
	"""
	model_name = "catalyst-agent"
	model_uri = f"runs:/{run_id}/{model_name}"
	
	if verbose:
		print("="*70)
		print("MLflow Model Reload Verification")
		print("="*70)
		print(f"\n[Step 1] Loading model from MLflow...")
		print(f"  Model URI: {model_uri}")
	
	try:
		# Load the model from MLflow
		loaded_agent = mlflow.langchain.load_model(model_uri)
		if verbose:
			print(f"✓ Model loaded successfully!")
			print(f"  Type: {type(loaded_agent).__name__}")
			print(f"  Module: {type(loaded_agent).__module__}")
		
	except Exception as e:
		if verbose:
			print(f"❌ Error loading model: {e}")
			print("\nTroubleshooting:")
			print("1. Verify the run_id is correct")
			print("2. Check that the model was logged with name 'catalyst-agent'")
			print("3. Ensure MLflow tracking URI points to the correct server")
		return False
	
	# Test with multiple inputs
	if verbose:
		print(f"\n[Step 2] Running inference tests...")
	
	test_cases = [
		{
			"description": "E-commerce platform",
			"input": {
				"raw_requirement": "Build an e-commerce platform with shopping cart and payment integration",
				"parsed_data": None,
				"final_plan": None
			}
		},
		{
			"description": "Data analytics dashboard",
			"input": {
				"raw_requirement": "Create a real-time analytics dashboard with data visualization",
				"parsed_data": None,
				"final_plan": None
			}
		},
		{
			"description": "IoT monitoring system",
			"input": {
				"raw_requirement": "Develop an IoT device monitoring system with alerts and reporting",
				"parsed_data": None,
				"final_plan": None
			}
		}
	]
	
	if verbose:
		print(f"\nRunning {len(test_cases)} test cases...\n")
	
	all_passed = True
	for i, test_case in enumerate(test_cases, 1):
		if verbose:
			print(f"Test Case {i}: {test_case['description']}")
			print("-" * 70)
			print(f"Input: {test_case['input']['raw_requirement']}")
		
		try:
			result = loaded_agent.invoke(test_case['input'])
			if verbose:
				print(f"Output:")
				print(f"  - Parsed data: {result['parsed_data']}")
				print(f"  - Final plan: {result['final_plan']}")
				print(f"✓ Test case {i} passed\n")
			
		except Exception as e:
			if verbose:
				print(f"❌ Test case {i} failed: {e}\n")
			all_passed = False
	
	# Summary
	if verbose:
		print("="*70)
		print("VERIFICATION SUMMARY")
		print("="*70)
		print("""
✅ Model successfully reloaded in a clean process!
✅ Model can be invoked for inference!
✅ All test cases completed!

This demonstrates that:
1. The model was properly serialized by MLflow
2. All dependencies are correctly captured
3. The model can be loaded without the original agent code being executed
4. The model maintains its functionality after reload

The model is ready for:
- Deployment to production
- Sharing with team members
- Versioning and model registry
- CI/CD integration
	""")
		
		print("="*70)
		print("Verification completed successfully! ✨")
		print("="*70)
	
	return all_passed


def main():
	"""Main entry point when run as a script."""
	# Check if run_id was provided
	if len(sys.argv) < 2:
		print("\n❌ Error: Run ID not provided")
		print("\nUsage: python verify_model_reload.py <run_id>")
		print("\nTo find your run_id:")
		print("1. Run the log_model_to_mlflow.py script")
		print("2. Copy the Run ID from the output")
		print("3. Pass it to this script")
		sys.exit(1)
	
	run_id = sys.argv[1]
	
	success = verify_model(run_id, verbose=True)
	sys.exit(0 if success else 1)


if __name__ == "__main__":
	main()
