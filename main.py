"""
Catalyst Agent Main - Simple LangGraph with MLflow Autologging
This demonstrates the minimal scaffolding with automatic MLflow tracing.
"""

import mlflow
from catalyst_agent.agent import create_agent

# =====================================
# 1. ENABLE MLFLOW AUTOLOGGING
# =====================================
# This single line enables automatic tracing for LangGraph
mlflow.langchain.autolog()

# Set experiment for organizing runs
mlflow.set_experiment("catalyst-agent-demo")

print("✓ MLflow autologging enabled")
experiment = mlflow.get_experiment_by_name('catalyst-agent-demo')
print(f"✓ Experiment set to: {experiment.name if experiment else 'catalyst-agent-demo'}")

# =====================================
# 2. CREATE THE AGENT
# =====================================
agent = create_agent()
print("✓ LangGraph agent created")

# =====================================
# 3. INVOKE THE AGENT (BASIC)
# =====================================
# This invocation will be automatically traced by MLflow
print("\n" + "="*50)
print("INVOCATION 1 - Basic usage with autolog")
print("="*50 + "\n")

input_data = {
    "raw_requirement": "Build a web application with user authentication",
    "parsed_data": None,
    "final_plan": None
}

result = agent.invoke(input_data)

print("\n" + "="*50)
print("AGENT RESULT:")
print("="*50)
print(f"Input requirement: {result['raw_requirement']}")
print(f"Parsed data: {result['parsed_data']}")
print(f"Final plan: {result['final_plan']}")
print("\n")

# =====================================
# 4. INVOKE THE AGENT (WITH DIFFERENT INPUT)
# =====================================
print("="*50)
print("INVOCATION 2 - Different requirement")
print("="*50 + "\n")

input_data2 = {
    "raw_requirement": "Create a REST API with authentication and database integration",
    "parsed_data": None,
    "final_plan": None
}

result2 = agent.invoke(input_data2)

print("\n" + "="*50)
print("AGENT RESULT:")
print("="*50)
print(f"Input requirement: {result2['raw_requirement']}")
print(f"Parsed data: {result2['parsed_data']}")
print(f"Final plan: {result2['final_plan']}")

# =====================================
# 5. VIEW TRACES
# =====================================
print("\n" + "="*50)
print("VIEWING TRACES")
print("="*50)
print("""
To view the traces:

1. Run MLflow UI:
   - Command: mlflow ui
   - Open: http://localhost:5000
   - Navigate to the "catalyst-agent-demo" experiment
   - Click on any run to see the trace visualization

2. The traces will show:
   - Agent workflow execution (START -> parse -> END)
   - Tool calls (parse_requirements)
   - Input/output for each step
   - Execution time for each component

Each invocation creates a separate trace that you can inspect.
""")

print("\n✓ Demo completed successfully!")
print("✓ Check MLflow UI to view detailed traces of both agent invocations")
