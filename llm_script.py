#!/usr/bin/env python3
import os
import sys
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> int:
    # 1) Загружаем переменные окружения (.env не обязателен)
    load_dotenv()

    parser = argparse.ArgumentParser(description="Первый запрос к LLM")
    parser.add_argument(
        "-q",
        "--query",
        default="Привет, бот! Назови столицу Франции.",
        help="Текст запроса к модели (по умолчанию — простой вопрос)",
    )
    args = parser.parse_args()

    # 2) Создаём клиента Gemini (v1.x)
    client = genai.Client()

    # Модель для использования
    MODEL_NAME = "gemini-2.5-flash"

    config = types.GenerateContentConfig(
        # Устанавливаем температуру. Например, 0.8 для более креативного ответа.
        temperature=0.8,
        # Вы также можете добавить max_output_tokens, top_p и top_k здесь.
    )

    # 3) Делаем минимальный запрос (Responses API)
    response = client.models.generate_content(
        model=MODEL_NAME, contents=args.query, config=config  # <--- Здесь передается конфигурация
    )

    # 4) Печатаем ответ (готовый текст)
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count

    # 5) (необязательно) печатаем число токенов, если оно доступно
    # 4. Вывод результата
    print("\n--- Ответ---")
    print(response.text)
    print("--------------------")

    print("Метаданные об использовании токенов:")
    print(f"  Токены во входном запросе (Prompt): {input_tokens}")
    print(f"  Токены в сгенерированном ответе (Response): {output_tokens}")
    print(f"  Всего использовано токенов (Total): {total_tokens}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
