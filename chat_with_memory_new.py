from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.llms import Ollama

# from langgraph.checkpoint.sqlite import SqliteSaver
# checkpointer = SqliteSaver.from_conn_string("database.db")


MODEL = "gemma3:4b"

llm = Ollama(model=MODEL, temperature=0, timeout=15)

agent = create_agent(
    llm,
    checkpointer=InMemorySaver(),  # Сохранение в памяти
    # checkpointer=SqliteSaver(conn=checkpointer),  # Сохранение в памяти
)

# Конфигурация с уникальным ID потока разговора
conv_1 = {"configurable": {"thread_id": "conversation_001"}}

question1 = "Привет! Меня зовут Алиса"
response1 = agent.invoke({"messages": [{"role": "user", "content": question1}]}, conv_1)
print("пользователь 1:", question1)
print("Бот:", response1["messages"][-1].content)

question2 = "Какое у меня имя?"
response2 = agent.invoke({"messages": [{"role": "user", "content": question2}]}, conv_1)
print("пользователь 1:", question2)
print("Бот:", response2["messages"][-1].content)

# Новый пользователь с другим thread_id
conv_2 = {"configurable": {"thread_id": "conversation_002"}}

question3 = "Какое у меня имя?"
response3 = agent.invoke({"messages": [{"role": "user", "content": question3}]}, conv_2)
print("пользователь 2:", question3)
print("Бот:", response3["messages"][-1].content)
