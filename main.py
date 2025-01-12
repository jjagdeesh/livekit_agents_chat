"""
To run this application:

First-time setup:
    1. Install Poetry if you haven't already:
       curl -sSL https://install.python-poetry.org | python3 -

    2. Install dependencies:
       poetry install

    3. Configure Poetry environment:
       poetry shell

    4. Set environment variables:
        export OPENAI_API_KEY=<your-openai-api-key>
       
Running the application:
    poetry run python main.py

Example chat:
    User: hi
    Agent:  Hello! How can I assist you with your airline inquiries today?
    User: When is my flight ?
    Agent:  Could you please provide me with your flight number or booking number? That way, I can find the information you need!
    User: My booking number is 123, what is my flight number ?
    Agent:  Your flight number is **AE123**. If you need more details about this flight, just let me know!
    User: Yes, when is leaving ?
    Agent:  Your flight AE123 is departing from **LAX** to **JFK** on **January 1, 2024**, at **10:00 AM**. The status is currently **On time**. If you need further assistance, feel free to ask!
    User: quit
    
Make sure you have Poetry installed and all dependencies are properly configured in pyproject.toml.
For more information about Poetry, visit: https://python-poetry.org/docs/
"""

import asyncio
import os

from src.chat_agent import ChatAgent    

async def main():
    new_agent = ChatAgent()
    new_agent.initialize_agent()

    while True:
        user = input("User: ")
        if user == "exit" or user == "quit":
            break
        response = await new_agent.chat_response(user)
        print("Agent: ", response)



if __name__ == "__main__":
    asyncio.run(main())
