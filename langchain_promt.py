from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser


template = "Объясни простыми словами, что такое {topic}."

prompt = PromptTemplate(input_variables=["topic"], template=template)

llm = Ollama(model="gemma3:4b", temperature=0)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "langchain"})

print(result)

