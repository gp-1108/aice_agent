GENERAL_SYSTEM_AND_USER_PROMPT = "<SYSTEM_MESSAGE>\n{system_message}\n\n<USER_MESSAGE>\n{user_message}</USER_MESSAGE>"

REQUIREMENTS_SYSTEM_PROMPT = """You are an expert requirements analyst. Your task is to analyze
raw requirement texts and extract key features, constraints, stakeholders, and success criteria.
Present the extracted information in a structured format."""

REQUIREMENTS_PROMPT = """Analyze the following raw requirement text and extract the key features,
constraints, stakeholders, and success criteria. Provide the information in a clear and structured manner.
Do not add any additional commentary or make assumptions beyond the provided text.

The description of the project is the following:

{raw_input}

Parse the output into the following sections:
1. Features: List the main features described in the requirements.
2. Constraints: List any constraints mentioned.
3. Stakeholders: Identify the stakeholders involved.
4. Success Criteria: Outline the criteria for success as described.
"""

COMPLEXITY_ESTIMATION_SYSTEM_PROMPT = """You are an expert project manager specializing in software development. Your task is to estimate the complexity of a feature
based on its requirements. Present your estimation in a structured format."""

COMPLEXITY_ESTIMATION_PROMPT = """Based on the following feature, estimate the complexity of the feature.
The feature is described below. The context of it is:
{raw_text}
To do this, consider the following factors:
- Difficulty Level: Classify the feature as Easy, Medium, Hard, or Very Hard.
- Estimated Days: Provide an estimate of the number of days required to implement the feature.
- Risks: Identify potential risks that could impact the implementation timeline.

Feature:
{parsed_requirements}
"""

TASK_GENERATION_PROMPT = """Based on the following estimated complexities for features and descriptions, generate a list of specific task names required to implement the feature.
For the feature, break it down into granular tasks with a meaningful title.

The feature is:
{parsed_feature}

Its estimated complexity is:
{estimated_complexity}

You must generate a list of task names (just the titles) so that they can fully cover the implementation of the feature.
Include all necessary tasks: setup, implementation, testing, and documentation."""

TASK_DETAILED_SYSTEM_PROMPT = """You are an expert project manager specializing in software development. Your task is to take a list of task titles and provide detailed descriptions, dependencies, and priority levels for ALL tasks at once."""

TASK_DETAILED_PROMPT = """Given the following list of task titles for a feature, provide detailed information for ALL tasks in a single structured output.

Feature context:
{feature_context}

Task titles:
{all_task_titles}

For each task, provide:
- Title: The exact title from the list above
- Description: A detailed description of what needs to be done
- Priority: Classify as Low, Medium, or High based on importance and blocking nature
- Dependencies: List the titles of other tasks (from the list above) that must be completed before this task can start. Use empty list [] if no dependencies.

Requirements:
1. Provide details for ALL tasks in the list
2. Ensure task titles exactly match those provided
3. Set realistic priorities based on task criticality
4. Identify dependencies accurately - only reference task titles from the provided list
5. Dependencies should form a logical order of execution

Generate the complete detailed task list now."""