<div id="top">

<!-- HEADER STYLE: ASCII -->
<div align="center">
<pre>
	__   ____  ______   ____  _      __ __  _____ ______       ____   ____    ___  ____   ______ 
   /  ] /    ||      | /    || |    |  |  |/ ___/|      |     /    | /    |  /  _]|    \ |      |
  /  / |  o  ||      ||  o  || |    |  |  (   \_ |      |    |  o  ||   __| /  [_ |  _  ||      |
 /  /  |     ||_|  |_||     || |___ |  ~  |\__  ||_|  |_|    |     ||  |  ||    _]|  |  ||_|  |_|
/   \_ |  _  |  |  |  |  _  ||     ||___, |/  \ |  |  |      |  _  ||  |_ ||   [_ |  |  |  |  |  
\     ||  |  |  |  |  |  |  ||     ||     |\    |  |  |      |  |  ||     ||     ||  |  |  |  |  
 \____||__|__|  |__|  |__|__||_____||____/  \___|  |__|      |__|__||___,_||_____||__|__|  |__|  
                                                                                                 
</pre>
</div>
<div align="center">

<em>Empower projects with seamless MLflow integration and tracking.</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=flat&logo=LangChain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=flat&logo=MLflow&logoColor=white" alt="MLflow">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">

</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

---

## Overview

**Catalyst Agent: Revolutionizing Project Development**

**Why Catalyst Agent?**

This project empowers developers with cutting-edge tools to enhance project planning and development processes. The core features include:

- **🚀 Enhanced Language Processing:** Seamlessly integrate langchain, langchain-openai, and langgraph libraries for advanced language processing capabilities.
- **💡 MLflow Management:** Efficiently manage ai agent experiments using mlflow version 3.5.0.
- **🔬 Catalyst Agent Functionality:** Demonstrate running, logging, and viewing MLflow models with ease.
- **📊 LangGraph Agent Workflow:** Generate structured workflows for project planning and development, optimizing efficiency and collaboration.

---

## Project Structure

```sh
└── /
    ├── catalyst_agent
    │   ├── __init__.py
    │   ├── agent.py
    │   ├── core.py
    │   ├── core_azure.py
    │   ├── output_structures.py
    │   ├── prompts.py
    │   ├── state.py
    │   └── tools.py
    ├── main.py
    ├── mlflow_logging
    │   ├── __init__.py
    │   ├── agent_model.py
    │   ├── log_model_to_mlflow.py
    │   └── verify_model_reload.py
    ├── mlflow_test_outputs
    │   ├── mlflow_test_20251021_234808_input1.json
    │   └── mlflow_test_20251021_235937_input2.json
    ├── mlruns
    │   ├── 861272485350255782
    ├── requirements.txt
```

### Project Index

<details open>
	<summary><b><code>/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Enhance language processing capabilities by integrating langchain, langchain-openai, and langgraph libraries<br>- Manage machine learning experiments with mlflow version 3.5.0.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/main.py'>main.py</a></b></td>
					<td style='padding: 8px;'>- Demonstrate the Catalyst Agents functionality by running, logging, and viewing MLflow models<br>- Interact with the LangGraph agent, save results to JSON, and manage MLflow experiments seamlessly<br>- Explore options to run the agent, log it to MLflow, and inspect or load logged models for enhanced project management and tracking capabilities.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- catalyst_agent Submodule -->
	<details>
		<summary><b>catalyst_agent</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ catalyst_agent</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/core_azure.py'>core_azure.py</a></b></td>
					<td style='padding: 8px;'>- Provide core utilities for the Catalyst Agent, including an Azure OpenAI chat model wrapper<br>- The <code>AzureLLM</code> class offers a convenient interface to interact with Azure OpenAI, handling configuration and offering simple methods for common operations like chatting, invoking the model with messages, streaming responses, and extracting structured data<br>- It serves as a crucial component for integrating Azure OpenAI capabilities within the agent.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/agent.py'>agent.py</a></b></td>
					<td style='padding: 8px;'>- Generate a LangGraph agent workflow that parses requirements, estimates complexity, generates tasks, creates acceptance criteria, and produces a final JSON output<br>- The workflow is orchestrated through defined nodes and edges, culminating in a compiled agent for project development planning.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/state.py'>state.py</a></b></td>
					<td style='padding: 8px;'>Define the state structure for the agent workflow, encompassing raw text, parsed requirements, estimated complexities, tasks, acceptance criteria, copilot prompts, and final JSON.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/tools.py'>tools.py</a></b></td>
					<td style='padding: 8px;'>- Parse requirements, estimate complexity, generate tasks, create acceptance criteria, and prompt for Copilot<br>- These tools leverage structured data to streamline project planning and development<br>- The code enhances collaboration and efficiency by automating key project management tasks.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/core.py'>core.py</a></b></td>
					<td style='padding: 8px;'>- Provide core utilities for the Catalyst Agent, including a wrapper for the OpenAI chat model<br>- Enables easy interaction with OpenAI, handling configuration and offering simple methods for common operations<br>- Supports chat interface, model invocation, streaming responses, and structured output<br>- Singleton instance ensures easy access project-wide, initializing only with available credentials.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/output_structures.py'>output_structures.py</a></b></td>
					<td style='padding: 8px;'>- Define structured data models for project features, requirements, tasks, and testing in the output_structures.py file<br>- Capture complexity estimates, task priorities, project phases, acceptance criteria, and Copilot prompts to facilitate project management and development processes.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/catalyst_agent/prompts.py'>prompts.py</a></b></td>
					<td style='padding: 8px;'>- Generate clear and structured prompts for analyzing requirements, estimating complexity, generating tasks, detailing tasks, creating acceptance criteria, and crafting Copilot prompts<br>- Tailor prompts to extract key project elements, estimate feature complexity, break down tasks, define detailed task information, set acceptance criteria, and generate concise prompts for AI coding assistants<br>- Ensure each prompt aligns with specific project needs and focuses on actionable guidance.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- mlflow_test_outputs Submodule -->
	<details>
		<summary><b>mlflow_test_outputs</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ mlflow_test_outputs</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/mlflow_test_outputs/mlflow_test_20251021_234808_input1.json'>mlflow_test_20251021_234808_input1.json</a></b></td>
					<td style='padding: 8px;'>- SummaryThe provided code file, located at <code>mlflow_test_outputs/mlflow_test_20251021_234808_input1.json</code>, outlines the requirements for building a web application with user authentication and a dashboard<br>- The project aims to create a web application with features such as user authentication and a dashboard interface<br>- The estimated complexities include potential challenges related to requirements ambiguity, authentication complexity, security and compliance considerations, integration dependencies, UI/UX scope underestimation, and deployment and environment configuration<br>- The code file serves as a guide for understanding the projects scope and potential challenges to be addressed during development.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/mlflow_test_outputs/mlflow_test_20251021_235937_input2.json'>mlflow_test_20251021_235937_input2.json</a></b></td>
					<td style='padding: 8px;'>- SummaryThe provided code file, located at <code>mlflow_test_outputs/mlflow_test_20251021_235937_input2.json</code>, outlines the requirements for creating a mobile app with push notifications and offline mode<br>- The file details the essential features such as a mobile application, push notifications for user engagement, and offline functionality for seamless operation without network connectivity<br>- It also highlights the estimated complexities involved in the development process, including platform scope considerations, backend requirements for push notifications, offline mode complexities, data storage decisions, and third-party dependencies integration challenges.This information serves as a foundational guide for architects and developers to understand the core functionalities and challenges associated with building the mobile app within the project architecture.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- mlflow_logging Submodule -->
	<details>
		<summary><b>mlflow_logging</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ mlflow_logging</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/mlflow_logging/verify_model_reload.py'>verify_model_reload.py</a></b></td>
					<td style='padding: 8px;'>- Verify MLflow model reload functionality for inference with clean process isolation<br>- Ensure successful model loading, invocation, and testing across various use cases<br>- Demonstrate model serialization, dependency capture, and readiness for deployment, sharing, versioning, and CI/CD integration.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/mlflow_logging/log_model_to_mlflow.py'>log_model_to_mlflow.py</a></b></td>
					<td style='padding: 8px;'>- Log Catalyst Agent to MLflow using models from code approach<br>- Enables seamless model logging, reloading, and inference<br>- Ensures compatibility, human-readable model definition, and avoids serialization issues<br>- Ideal for LangGraph agents with custom nodes<br>- Detailed documentation and instructions provided for model verification, deployment, and viewing in MLflow UI.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/mlflow_logging/agent_model.py'>agent_model.py</a></b></td>
					<td style='padding: 8px;'>Define the MLflow model by setting the Catalyst Agent, ensuring seamless logging of the production model.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip

### Installation

Build  from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/gp-1108/aice_agent.git
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd aice_agent
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![pip][pip-shield]][pip-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [pip-shield]: https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white -->
	<!-- [pip-link]: https://pypi.org/project/pip/ -->

	**Using [pip](https://pypi.org/project/pip/):**

	```sh
	❯ pip install -r requirements.txt
	```

### Usage

Run the project with:

**Using [pip](https://pypi.org/project/pip/):**
```sh
python main.py
```
If you want to log the model to MLflow, run:
```sh
python -m mlflow_logging.log_model_to_mlflow
```

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square

---
