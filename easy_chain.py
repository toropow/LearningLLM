from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableSequence


MODEL = "gemma3:4b"

llm = Ollama(model=MODEL, temperature=0, timeout=15)

promt = ChatPromptTemplate.from_template("Переведи на английский: {text}")

chain = RunnableSequence(first=promt, last=llm)

result = chain.invoke({"text": "Доброе утро"})

print(result)
