import json
import re
from typing import Dict, List, Any, Union
from langchain_openai import AzureChatOpenAI as ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

def create_json_validator_agent():
    llm = ChatOpenAI(
        azure_deployment="ClarityChain",  # or your deployment
        api_version="2024-08-01-preview",
        temperature=0,
    )

    # The JSON validator agent does not need any external tools, its task is purely text processing and JSON validation using the LLM.
    tools = []

    json_validator_template = """You are an expert JSON validation and extraction agent. Your task is to extract a single, valid JSON object or array from the provided text.

The text may contain conversational elements, markdown, or other non-JSON content. You must identify and extract ONLY the JSON.

If the text contains a valid JSON object or array, return it as your Final Answer. If multiple JSON objects/arrays are present, extract the most relevant one based on the context of data analysis. If no valid JSON is found, return an empty JSON object: {{}}.

Text to process:
{text_to_validate}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. After each thought, you must provide an Action.
Action: The action to take, should be one of [{tool_names}]. The `Action:` line must contain *only* the tool name, without any extra characters, markdown, or formatting.
Action Input: the input to the action. The `Action Input:` line must contain *only* the input for the tool, and it must be on a single line.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have extracted and validated the JSON and will now return it.
Final Answer: the extracted and validated JSON object or array.

Begin!

Thought: I must carefully examine the provided text to identify and extract a valid JSON object or array. I will ignore any conversational or non-JSON content.
{agent_scratchpad}

TOOLS:
------
{tools}

TOOLS:
------
{tools}
"""

    prompt = PromptTemplate(
        template=json_validator_template,
        input_variables=["text_to_validate", "agent_scratchpad"],
        partial_variables={
        },
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor