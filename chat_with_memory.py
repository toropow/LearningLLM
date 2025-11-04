from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = Ollama(model="gemma3:4b", temperature=0)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Функция для получения или создания истории по session_id.
    Каждый пользователь/разговор имеет свой session_id.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты асистент, который хорошо разбирается в {ability}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    # добавляем нашу функцию для сохранения истории переписки
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Первый запрос
response1 = chain_with_history.invoke(
    {"ability": "математика", "question": "Что такое косинус?"}, config={"configurable": {"session_id": "user_123"}}
)
print("Бот:", response1)
print("====================================")

# Второй запрос (бот помнит, что говорилось о математике и косинусе)
response2 = chain_with_history.invoke(
    {"ability": "математика", "question": "А как он связан с синусом?"},
    config={"configurable": {"session_id": "user_123"}},
)
print("Бот:", response2)
print("====================================")

# Третий пользователь – отдельная история
response3 = chain_with_history.invoke(
    {"ability": "история", "question": "Кто был Александр Великий?"},
    config={"configurable": {"session_id": "user_456"}},
)
print("Бот:", response3)
print("====================================")
