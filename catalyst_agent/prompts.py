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