from dotenv import load_dotenv

# Обновленные импорты для LangChain Core и Community
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Импорты для управления историей
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging

logging.basicConfig(
    filename="chat_session.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("=== New session ===")


# Глобальное хранилище истории
# В продакшене это должна быть база данных (Redis, DynamoDB и т.п.)
# Для примера используем in-memory словарь.
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Функция для получения истории диалога по ID сессии."""
    if session_id not in store:
        # Для простого примера используем ChatMessageHistory (in-memory)
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def setup_environment():
    """Загрузка переменных окружения."""
    load_dotenv()


def create_chat_runnable(model_name: str, system_message: str):
    """
    Создает цепочку (Runnable) с промптом и моделью.
    """
    # 1. Создание модели
    chat_model = Ollama(model=model_name, temperature=0, timeout=15)

    # 2. Создание промпта
    # MessagesPlaceholder необходим для вставки истории диалога
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),  # Сюда будет вставляться история
            ("human", "{input}"),  # Сюда будет вставляться новый запрос пользователя
        ]
    )

    # 3. Создание базовой цепочки с использованием LCEL
    # RunnablePassthrough передает входные данные дальше без изменений.
    # Это позволяет передать {input} и {history} в промпт.
    # | (pipe) — оператор композиции в LCEL
    chain = prompt | chat_model | StrOutputParser()

    # 4. Оборачивание цепочки в RunnableWithMessageHistory
    # Это добавляет логику автоматического извлечения/сохранения истории.
    # input_messages_key: ключ во входном словаре, который содержит новый запрос
    # history_messages_key: ключ, в который будет вставлена история
    runnable_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return runnable_with_history


def chat_loop(runnable_with_history):
    """
    Основной цикл консольного чата.
    """
    # Жестко задаем ID сессии для консольного приложения
    # В реальном приложении это был бы ID пользователя/чата
    session_id = "console-user-123"

    while True:
        try:
            user_text = input("Вы: ")
            user_text = user_text.strip()
            logging.info(f"User: {user_text}")
        except (KeyboardInterrupt, EOFError):
            print("\nБот: Завершение работы.")
            break

        user_text = user_text.strip()
        if user_text == "":
            continue

        cmd = user_text.lower()
        if cmd in ("выход", "стоп", "конец"):
            print("Бот: До свидания!")
            logging.info("User initiated exit. Session ended.")
            break
        if cmd in ("сброс",):
            # Очистка истории для текущей сессии
            if session_id in store:
                store[session_id].clear()
                print("Бот: Контекст диалога очищен.")
                logging.info("Context clean")
            else:
                print("Бот: Контекст диалога еще не был создан.")
                logging.info("Context doesn't create yet")
            continue

        try:
            # .invoke требует словарь аргументов:
            # - 'input' (новый запрос)
            # - 'configurable' (словарь, содержащий session_id для истории)
            reply_text = runnable_with_history.invoke(
                {"input": user_text},
                config={"configurable": {"session_id": session_id}},
            )
            reply_text = reply_text.strip()

        except Exception as e:
            # В отличие от ConversationChain, здесь ответ — это уже строка,
            # если в конце стоит StrOutputParser
            logging.error(f"Error: {e}")
            print(f"Бот: [Ошибка] {e}")
            continue

        print(f"Бот: {reply_text}")
        logging.info(f"Bot: {reply_text}")


def main():
    # Настройка доступа
    setup_environment()

    # Системное сообщение
    SYSTEM_MESSAGE = "Ты грубый, но веселый ассистент"
    MODEL_NAME = "gemma3:4b"

    # 1. Создание runnable с историей
    conversation_runnable = create_chat_runnable(
        model_name=MODEL_NAME, system_message=SYSTEM_MESSAGE.format(model_name=MODEL_NAME)
    )

    print("Привет! Я консольный бот. Для выхода напишите «выход», для сброса контекста — «сброс».")
    # 2. Запуск цикла
    chat_loop(conversation_runnable)


if __name__ == "__main__":
    main()
