"""Executor that calls a remote LangGraph agent.

This module demonstrates how to integrate a LangGraph agent running on a web server
as the task executor. The agent is invoked via HTTP using direct REST API calls.

Usage:
    Ensure the LangGraph agent is running at http://localhost:2024 before executing.
    Start the agent with: langgraph dev

Environment Variables:
    LANGGRAPH_URL: Optional. Override the default server URL (default: http://localhost:2024).
    LANGGRAPH_ASSISTANT_ID: Optional. Override the assistant ID (default: agent).
    LANGGRAPH_AUTH_TOKEN: Required. Bearer token for LangGraph server authentication.
"""

import json
import os

import httpx

# Configuration (can be overridden via environment variables)
LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL", "http://localhost:2024")
ASSISTANT_ID = os.environ.get("LANGGRAPH_ASSISTANT_ID", "agent")
AUTH_TOKEN = os.environ.get("LANGGRAPH_AUTH_TOKEN", "")


def executor(messages):
    """Execute LLM calls via a remote LangGraph agent.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Typically includes a system message and user message.

    Returns:
        String response from the LangGraph agent.

    Raises:
        ConnectionError: If the LangGraph server is not reachable.
        Exception: If the agent returns an error or unexpected response.
    """
    try:
        # Format input for the LangGraph agent
        input_data = {
            "messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
        }

        # Make streaming request to the LangGraph runs endpoint
        request_body = {
            "assistant_id": ASSISTANT_ID,
            "input": input_data,
            "stream_mode": "messages-tuple",
        }

        final_content = ""

        # Build headers with optional auth
        headers = {"Content-Type": "application/json"}
        if AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{LANGGRAPH_URL}/runs/stream",
                json=request_body,
                headers=headers,
            ) as response:
                response.raise_for_status()

                # Process Server-Sent Events (SSE) stream
                # SSE data can span multiple lines, so accumulate until we get valid JSON
                current_event = ""
                data_buffer = []

                for line in response.iter_lines():
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                        data_buffer = []
                    elif line.startswith("data:"):
                        data_buffer.append(line[5:])
                    elif not line and data_buffer:
                        # Empty line marks end of event, parse accumulated data
                        if current_event == "messages":
                            try:
                                data_str = "".join(data_buffer)
                                data = json.loads(data_str)
                                # Format: [message_dict, metadata_dict]
                                if isinstance(data, list) and len(data) >= 1:
                                    message = data[0]
                                    if isinstance(message, dict):
                                        msg_type = message.get("type", "")
                                        if msg_type == "ai":
                                            content = message.get("content", "")
                                            if content and isinstance(content, str):
                                                final_content = content
                            except json.JSONDecodeError:
                                pass
                        data_buffer = []

        if not final_content:
            raise ValueError("No response content received from LangGraph agent")

        return final_content

    except httpx.ConnectError as e:
        raise ConnectionError(
            f"Failed to connect to LangGraph server at {LANGGRAPH_URL}. "
            f"Ensure the server is running with 'langgraph dev'. Error: {e}"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LangGraph server returned error: {e.response.status_code}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error calling LangGraph agent: {e}. "
            f"Check that the agent is configured correctly."
        ) from e
