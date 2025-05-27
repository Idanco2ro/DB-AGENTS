import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json

#Exemple dummy function hard coded to return the same weather
#In production, this could be your backend API or as external API
def get_current_weather(location, unit="Fahrenheit"):
   """Get the current weather for a given location."""
   if "tokio" in location.lower():
       return json.dumps({"location": "Tokyo", "temperature": 10, "unit": unit})
   elif "san francisco" in location.lower():
       return json.dumps({"location": "San Francisco", "temperature": 72, "unit": unit})
   elif "paris" in location.lower():
       return json.dumps({"location": "Paris", "temperature": 22, "unit": unit})
   else:
       return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    #Step 1: send the conversation and available function to the model
    messages = [
        {
            "role": "user",
            "content": "What is the weather in San Francisco, Tokyo, and Paris?",
        }
    ]
    
    #Define the available functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for.",
                        },
                        "unit": {
                            "type": "string",
                            "description": 'The unit of temperature. Can be either "Celsius" or "Fahrenheit".',
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    
    # ✅ Load environment variables before using them
    load_dotenv()

    # ✅ Add this to initialize the OpenAI client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Automatically choose the best tool
    )

    response_message = response.choices[0].message
    # print(response_message.model_dump_json(indent=2))
    # print("tool calls:", response_message.tool_calls)
    
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather,
        }

    messages.append(response_message)

    #Step 4: send the info for each function call and function response
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        function_response = available_functions[function_name](
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    second_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    return second_response
    
    
    
# ✅ Call the function to run the conversation
if __name__ == "__main__":
    result = run_conversation()
    print(result.model_dump_json(indent=2))


