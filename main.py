from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from venv.vector import retriever

model = OllamaLLM(model="llama3.2")

template = """"
you are an expert in answering questions about a pizza resturant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("/n/n--------------------")
    question = input("Enter your question about pizza places: (q to quit) ")
    if question.lower() == "q":
        break

#Goes to vector database to get relevant reviews and prints answer
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
