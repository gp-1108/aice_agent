"""
Core utilities for the Catalyst Agent.

This module provides shared resources and utilities used across the agent,
including the Azure OpenAI chat model wrapper.
"""

import os
from typing import List, Optional, Union, Type, Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel
load_dotenv()


class AzureLLM:
    """
    Wrapper class for Azure OpenAI chat model.
    
    This class provides a convenient interface to interact with Azure OpenAI,
    handling configuration and providing simple methods for common operations.
    
    Example:
        >>> llm = AzureLLM()
        >>> response = llm.chat("Tell me about Paris")
        >>> print(response)
        
        >>> # With system message
        >>> response = llm.chat(
        ...     "What should I see in Paris?",
        ...     system_message="You are a travel guide."
        ... )
        
        >>> # Advanced usage with message history
        >>> messages = [
        ...     SystemMessage(content="You are a helpful assistant."),
        ...     HumanMessage(content="Hello!"),
        ... ]
        >>> response = llm.invoke(messages)
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Catalyst LLM wrapper.
        
        Args:
            endpoint: Azure OpenAI endpoint URL. Defaults to environment variable 
                     AZURE_OPENAI_ENDPOINT or class default.
            deployment: Azure deployment name. Defaults to environment variable 
                       AZURE_OPENAI_DEPLOYMENT or class default.
            api_key: Azure API key. Defaults to environment variable 
                    AZURE_OPENAI_API_KEY or class default.
            api_version: Azure API version. Defaults to class default.
            temperature: Sampling temperature (0-1). Default is 0.7.
            **kwargs: Additional arguments passed to AzureChatOpenAI.
        """
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "")
        self.temperature = temperature
        
        # Initialize the Azure Chat model
        self.model = AzureChatOpenAI(
            azure_endpoint=self.endpoint,
            azure_deployment=self.deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            temperature=temperature,
            **kwargs
        )
    
    def chat(
        self, 
        user_message: str, 
        system_message: Optional[str] = None,
        message_history: Optional[List[BaseMessage]] = None
    ) -> str:
        """
        Simple chat interface that returns just the content string.
        
        Args:
            user_message: The user's message/question.
            system_message: Optional system message to set context/behavior.
            message_history: Optional list of previous messages for context.
            
        Returns:
            The assistant's response as a string.
            
        Example:
            >>> llm = AzureLLM()
            >>> response = llm.chat("What is Python?")
            >>> print(response)
        """
        messages = []
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # Add user message
        messages.append(HumanMessage(content=user_message))
        
        # Get response
        response = self.model.invoke(messages)
        return response.content
    
    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Invoke the model with a list of messages.
        
        This provides direct access to the underlying model's invoke method
        for advanced usage.
        
        Args:
            messages: List of message objects (SystemMessage, HumanMessage, etc.)
            
        Returns:
            AIMessage object with the full response.
            
        Example:
            >>> llm = AzureLLM()
            >>> messages = [
            ...     SystemMessage(content="You are a helpful assistant."),
            ...     HumanMessage(content="Hello!")
            ... ]
            >>> response = llm.invoke(messages)
            >>> print(response.content)
        """
        return self.model.invoke(messages)
    
    def stream(self, messages: List[BaseMessage]):
        """
        Stream the model's response token by token.
        
        Args:
            messages: List of message objects.
            
        Yields:
            Response chunks as they're generated.
            
        Example:
            >>> llm = AzureLLM()
            >>> messages = [HumanMessage(content="Tell me a story")]
            >>> for chunk in llm.stream(messages):
            ...     print(chunk.content, end="", flush=True)
        """
        return self.model.stream(messages)
    
    def __call__(self, user_message: str, **kwargs) -> str:
        """
        Allow the instance to be called directly like a function.
        
        Args:
            user_message: The user's message.
            **kwargs: Additional arguments passed to chat().
            
        Returns:
            The assistant's response as a string.
            
        Example:
            >>> llm = AzureLLM()
            >>> response = llm("What is machine learning?")
            >>> print(response)
        """
        return self.chat(user_message, **kwargs)
    
    def get_model(self) -> AzureChatOpenAI:
        """
        Get the underlying AzureChatOpenAI model instance.
        
        Useful when you need to pass the model to LangChain chains or agents.
        
        Returns:
            The AzureChatOpenAI instance.
            
        Example:
            >>> llm = AzureLLM()
            >>> model = llm.get_model()
            >>> # Use model in a chain
            >>> chain = prompt | model | output_parser
        """
        return self.model
    
    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        **kwargs
    ) -> Runnable:
        """
        Get a version of the model that returns structured output.
        
        This method returns a Runnable that will output objects matching the
        provided schema instead of plain text. The schema can be a Pydantic
        model or a JSON schema dictionary.
        
        Args:
            schema: Either a Pydantic BaseModel class or a JSON schema dict
                   that specifies the structure of the desired output.
            **kwargs: Additional arguments passed to with_structured_output().
            
        Returns:
            A Runnable that outputs structured data matching the schema.
            
        Example with Pydantic:
            >>> from pydantic import BaseModel, Field
            >>> 
            >>> class Joke(BaseModel):
            ...     setup: str = Field(description="The setup of the joke")
            ...     punchline: str = Field(description="The punchline")
            ...     rating: int = Field(description="Funny rating 1-10")
            >>> 
            >>> llm = AzureLLM()
            >>> structured_llm = llm.with_structured_output(Joke)
            >>> joke = structured_llm.invoke("Tell me a joke about cats")
            >>> print(f"Setup: {joke.setup}")
            >>> print(f"Punchline: {joke.punchline}")
            
        Example with TypedDict/JSON Schema:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "integer"}
            ...     },
            ...     "required": ["name", "age"]
            ... }
            >>> structured_llm = llm.with_structured_output(schema)
            >>> result = structured_llm.invoke("John is 30 years old")
            >>> print(result)  # {'name': 'John', 'age': 30}
        """
        return self.model.with_structured_output(schema, **kwargs)
    
    def extract(
        self,
        text: str,
        schema: Type[BaseModel],
        system_message: Optional[str] = None
    ) -> BaseModel:
        """
        Convenience method to extract structured data from text.
        
        This wraps with_structured_output for quick data extraction tasks.
        
        Args:
            text: The text to extract information from.
            schema: A Pydantic model defining the structure to extract.
            system_message: Optional instruction for the extraction.
            
        Returns:
            An instance of the schema with extracted data.
            
        Example:
            >>> from pydantic import BaseModel
            >>> 
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            ...     occupation: str
            >>> 
            >>> llm = AzureLLM()
            >>> person = llm.extract(
            ...     "Alice is a 28 year old software engineer",
            ...     Person
            ... )
            >>> print(f"{person.name} is {person.age} years old")
        """
        structured_model = self.with_structured_output(schema)
        
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=text))
        
        return structured_model.invoke(messages)


# Singleton instance for easy access throughout the project
# Only initialize if credentials are available
_llm_instance = None

def get_llm() -> AzureLLM:
    """
    Get or create the singleton LLM instance.
    
    Returns:
        The AzureLLM singleton instance.
        
    Example:
        >>> from catalyst_agent.core import get_llm
        >>> llm = get_llm()
        >>> response = llm.chat("Hello!")
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = AzureLLM()
    return _llm_instance


# Convenience function for quick one-off calls
def ask_llm(question: str, system_prompt: Optional[str] = None) -> str:
    """
    Convenience function for quick LLM queries.
    
    Args:
        question: The question or prompt.
        system_prompt: Optional system message.
        
    Returns:
        The LLM's response as a string.
        
    Example:
        >>> from catalyst_agent.core import ask_llm
        >>> answer = ask_llm("What is the capital of France?")
        >>> print(answer)
    """
    return get_llm().chat(question, system_message=system_prompt)