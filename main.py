from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectorSearch import retriever

model = OllamaLLM(model="gemma3:1b")

template = """
You are an expert in answering question about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is a question about the restaurant: {question}
"""

prompt = ChatPromptTemplate.from_template(template) # It takes variables ({reviews} and {question}) and formats them into a complete prompt string for the language model.
chain = prompt | model # This creates a chain that combines the prompt and the model (the model will receive the prompt's output as input).

while True:
    print("\n\n---------------------------------------")
    user_input = input("Enter your reviews (or type 'exit' to quit): ")
    print("\n\n")
    if user_input.lower() == "exit":
        break
    reviews = retriever.invoke(user_input)
    result = chain.invoke({
        "reviews": [reviews],
        "question": [user_input]
    })

    print(result)