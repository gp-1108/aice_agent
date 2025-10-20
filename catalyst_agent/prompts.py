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
