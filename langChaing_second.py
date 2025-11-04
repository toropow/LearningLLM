from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

response = llm.invoke("Какой сегодня день недели?")
print(response.content)