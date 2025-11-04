from langchain_community.llms import Ollama

llm = Ollama(model="gemma3:4b")

prompt = "Напиши короткое стихотворение о программировании на Python."
print(f"Промпт: {prompt}\n")

# Получение ответа
response = llm.invoke(prompt)

print("--- Ответ модели ---")
print(response)
print("--------------------")