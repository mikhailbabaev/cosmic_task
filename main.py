from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timezone, timedelta

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a smart assistant who can call tools.\n"
         "Currently, you have a tool called get_current_time, which you can use to find out the current time and date.\n"
         "Here is an example of what you will get from this tool: Example → {\"utc\": \"2025‑05‑21T06:42:00Z\"}\n"
         "Не делай цепочки рассуждений"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)





def get_current_time(_: str) -> str:
    moscow_tz = timezone(timedelta(hours=3))
    now = datetime.now(moscow_tz)
    return now.strftime("Current time: %Y-%m-%d %H:%M:%S %Z")


tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Returns the current UTC time in ISO-8601 format",
    )
]

llm = OllamaLLM(model="llama3.2")
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prompt": prompt},
    max_iterations=3,
    verbose=True
)

response = agent.invoke({"input": "What time is it?"})
print(response["output"])


