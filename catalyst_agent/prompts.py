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

ACCEPTANCE_CRITERIA_SYSTEM_PROMPT = """You are an expert QA engineer and test architect specializing in software development. Your task is to create clear, testable acceptance criteria and comprehensive test descriptions for software tasks."""

ACCEPTANCE_CRITERIA_PROMPT = """Given a feature and its associated tasks, create acceptance criteria and test descriptions for each task.

Feature context:
{feature_context}

Tasks:
{tasks_list}

For EACH task, provide:

1. Acceptance Criteria (multiple criteria in Given/When/Then format):
   - Given: The initial context or precondition
   - When: The action or event that occurs
   - Then: The expected outcome or result
   
2. Unit Tests: Descriptions of unit tests needed
   - test_name: A descriptive name for the test
   - test_type: "unit"
   - description: What the test validates
   
3. Integration Tests: Descriptions of integration tests (if relevant for the task)
   - test_name: A descriptive name for the test
   - test_type: "integration"
   - description: What the test validates across components

Requirements:
1. Create 2-4 acceptance criteria per task in Given/When/Then format
2. Be specific and testable - avoid vague criteria
3. Include relevant unit tests for each task (at least 2-5 per task)
4. Include integration tests only when the task involves multiple components or external systems
5. Ensure test names are descriptive and follow naming conventions
6. Cover positive cases, negative cases, and edge cases
7. Ensure task_title exactly matches the task title from the list above

Generate acceptance criteria and tests for ALL tasks now."""

COPILOT_PROMPT_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in creating concise, high-signal prompts for AI coding assistants like GitHub Copilot and Claude. Your task is to distill task requirements and acceptance criteria into clear, actionable prompts."""

COPILOT_PROMPT_GENERATION = """Given a feature, its tasks, and their acceptance criteria, create concise, high-signal prompts suitable for GitHub Copilot or Claude for EACH task.

Feature context:
{feature_context}

Tasks with acceptance criteria:
{tasks_with_criteria}

For EACH task, create a prompt that:

1. Starts with a clear action verb (e.g., "Implement", "Create", "Build", "Add")
2. Clearly states what needs to be built
3. Includes key requirements from the task description
4. Incorporates critical acceptance criteria (Given/When/Then)
5. Mentions important test scenarios
6. Is concise (2-4 sentences, max 150 words)
7. Uses technical language appropriate for the context
8. Focuses on WHAT to build and HOW to validate it

Format guidelines:
- Be direct and specific
- Include success criteria inline
- Mention edge cases if critical
- Reference test requirements briefly
- Avoid unnecessary fluff or explanations
- Use concrete technical terms

Example format:
"Implement [functionality] that [does X]. It should handle [scenario 1] and [scenario 2]. 
Ensure [acceptance criterion]. Include unit tests for [test scenarios]."

Requirements:
1. Generate ONE prompt per task
2. Ensure task_title exactly matches the task title from the list
3. Keep prompts focused and actionable
4. Include only the most critical information
5. Make prompts immediately usable by an AI coding assistant

Generate Copilot prompts for ALL tasks now."""