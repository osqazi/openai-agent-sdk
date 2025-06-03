import os
import asyncio
from agents import Agent, Runner,OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_api_url = os.getenv("GEMINI_API_URL")
gemini_api_model = os.getenv("GEMINI_API_MODEL")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=gemini_api_url,
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model='gemini-2.0-flash',
)


config =  RunConfig(
    model=model,
    tracing_disabled=True
)
 

agent: Agent = Agent(
    name="Assistant",
    instructions="An Agent who is a helpful assistant.",
    model=model,  
)

async def main():
    prompt = ''
    while prompt.lower() != "x":
        prompt = input("Ask Anything (write X to quit): ")
        result = await Runner.run(
            agent,
            input=prompt,
            run_config=config,
        )
        print("\nCalling Agent...\n")
        print(result.final_output)

asyncio.run(main())