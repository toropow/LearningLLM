from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7,
    max_output_tokens=512,  # лимит токенов в ответе
    timeout=60,  # таймаут запроса в секундах)
)
response = llm.invoke("Какой сегодня день недели?")
print(response.content)