from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
import requests
import os

# Load environment variables (e.g., API keys)
load_dotenv()


async def main():
    # Define the task for browser-use
    task = "Go to youtube, search for latest deepseek video, and play it in fullscreen mode"

    # Initialize the browser-use agent
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o"),  # Use GPT-4 or another model
    )

    # Run the browser automation task
    result = await agent.run()
    print("Browser-use result:", result)

    # Convert the result to a JSON-serializable format
    def serialize_result(result):
        if hasattr(result, "all_results") and hasattr(result, "all_model_outputs"):
            return {
                "all_results": [
                    {
                        "is_done": r.is_done,
                        "extracted_content": r.extracted_content,
                        "error": r.error,
                        "include_in_memory": r.include_in_memory,
                    }
                    for r in result.all_results
                ],
                "all_model_outputs": result.all_model_outputs,
            }
        # Fallback: convert to string if serialization fails
        return str(result)

    serialized_result = serialize_result(result)

    # Integrate with DeepSeek R1 (example: send result to DeepSeek R1 via API)
    # Replace with actual URL
    deepseek_url = os.getenv(
        "DEEPSEEK_API_URL", "http://deepseek-r1-api-endpoint.com/data")
    # Load API key from environment variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set in the environment variables.")

    headers = {
        # Use Bearer token for authentication
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json",
    }

    payload = {"data": serialized_result}

    try:
        response = requests.post(deepseek_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print("DeepSeek R1 response:", response.json())
    except requests.exceptions.ConnectionError as e:
        print(f"Failed to connect to DeepSeek R1: {e}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


# Run the script
asyncio.run(main())
