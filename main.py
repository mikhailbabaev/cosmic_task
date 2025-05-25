from datetime import datetime, timezone
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


def get_current_time(_: str) -> str:
    """Return the current UTC time in ISO‑8601 format.
    Example → 2025‑05‑21T06:42:00Z"""
    now = datetime.now(timezone.utc)
    return now.isoformat()


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Returns the current UTC time in ISO-8601 format",
    )
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a smart assistant who can call tools.\n"
                "You can determine the current time by a tool called get_current_time.\n"
            ),
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = OllamaLLM(model="llama3.2")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"prompt": prompt},
    max_iterations=5,
    verbose=True,
    handle_parsing_errors=True,
)


def chatbot_node(state: State) -> State:
    user_message = next((m for m in reversed(state["messages"]) if m["role"] == "user"), None)
    if user_message is None:
        return {"messages": [AIMessage(content="No user input found.").model_dump()]}

    input_text = user_message["content"]
    response = agent.invoke({"input": input_text})
    output_text = response["output"]
    ai_message = AIMessage(content=output_text).model_dump()
    return {"messages": [ai_message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


if __name__ == "__main__":
    initial_state = {
        "messages": [
            {"role": "user", "content": "What time is it?"}
        ]
    }

    new_state = chatbot_node(initial_state)
    print("Ответ агента:")
    print(new_state["messages"][0]["content"])
