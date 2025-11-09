import os
from dotenv import load_dotenv

load_dotenv()

import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


# Создаём класс для CLI-бота
class cli_bot():
    def __init__(self, model_name, system_prompt="Ты полезный ассистент."):
        # Создаём модель
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=15
        )

        # Создаём Хранилище истории
        self.store = {}

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),  # Добавим возможность менять системный промпт
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Создаём цепочку
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,  # Цепочка с историей
            self.get_session_history,  # метод для получения истории
            input_messages_key="question",  # ключ для вопроса
            history_messages_key="history",  # ключ для истории
        )

        # Настраиваем логирование в файл
        logging.basicConfig(
            filename="chat_session.log", level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            encoding="utf-8"
        )

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def __call__(self, session_id):
        print(
            "Чат-бот запущен! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        logging.info("=== New session ===")
        while True:
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            logging.info(f"User: {user_text}")
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logging.info("Пользователь завершил сессию. Сессия окончена.")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                logging.info("Пользователь сбросил контекст.")
                continue

            try:
                responce = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": session_id}}
                )
            except Exception as e:
                # Логируем и выводим ошибку, продолжаем чат
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                continue

            # Форматируем и выводим ответ
            bot_reply = responce.content.strip()
            logging.info(f"Bot: {bot_reply}")
            print(f"Бот: {bot_reply}")


if __name__ == "__main__":
    model = os.getenv("OPENAI_API_MODEL", "gpt-5")
    system_prompt = '''Ты полезный ассистент. Ты всегда дружелёбен и вежлив. Отвечай подробно и по существу.'''

    bot = cli_bot(
        model_name=model,
        system_prompt=system_prompt
    )
    bot("user_123")