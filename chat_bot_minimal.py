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
        ("system", "Ты асистент, который хорошо разбирается во всем"),
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


if __name__ == "__main__":
    print("Начинаем диалог с ботом (для выхода введите 'выход')")

    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ("выход", "exit", "quit"):
            print("Бот: До свидания!")
            break
        try:
            bot_reply = chain_with_history.invoke(
                {"question": user_input}, config={"configurable": {"session_id": "user"}}
            )
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            continue
        print(f"Бот: {bot_reply}")
