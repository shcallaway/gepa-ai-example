"""Executor with a mock tool-using agent.

This demonstrates how to capture tool calls for evaluation.
Replace the mock agent with your actual tool-using agent.
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm


# Define available tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations. Use for any math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2' or '15 * 0.15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Paris' or 'New York'",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool and return the result. Replace with real implementations."""
    if name == "calculator":
        try:
            # Simple safe eval for basic math
            expr = args.get("expression", "")
            # Only allow digits, operators, parentheses, and decimal points
            if re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expr):
                result = eval(expr)  # noqa: S307
                return str(result)
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {e}"

    elif name == "weather":
        # Mock weather data
        location = args.get("location", "").lower()
        mock_weather = {
            "paris": "22°C and sunny",
            "london": "15°C and cloudy",
            "new york": "28°C and humid",
            "tokyo": "26°C and partly cloudy",
        }
        return mock_weather.get(location, "18°C and clear")

    elif name == "search":
        # Mock search results
        query = args.get("query", "").lower()
        if "capital" in query and "france" in query:
            return "Paris is the capital of France."
        elif "population" in query and "tokyo" in query:
            return "Tokyo has a population of approximately 14 million people."
        elif "python" in query:
            return "Python is a high-level programming language created by Guido van Rossum."
        return f"Search results for: {query}"

    return f"Unknown tool: {name}"


def executor(messages: list[dict[str, str]]) -> str:
    """Execute the agent with tool calling and return structured output.

    Returns a JSON string containing:
    - answer: The final response
    - tool_calls: List of tools called with args and results
    """
    tool_calls_log: list[dict[str, Any]] = []

    # Build conversation with tools
    conversation = list(messages)

    # Allow up to 5 tool-calling rounds
    for _ in range(5):
        response = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=conversation,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Check if the model wants to call tools
        if assistant_message.tool_calls:
            # Add assistant message to conversation
            conversation.append(assistant_message.model_dump())

            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                result = execute_tool(func_name, func_args)

                # Log the tool call
                tool_calls_log.append({
                    "name": func_name,
                    "args": func_args,
                    "result": result,
                })

                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # No more tool calls, return final answer
            final_answer = assistant_message.content or ""
            return json.dumps({
                "answer": final_answer,
                "tool_calls": tool_calls_log,
            })

    # Max iterations reached, return what we have
    return json.dumps({
        "answer": "Unable to complete the request.",
        "tool_calls": tool_calls_log,
    })
