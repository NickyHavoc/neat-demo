from utils import LLMWrapper


messages = [
    {
        "role": "system",
        "content": "You are a Albert Einstein. When asked who you are, say Einstein."
    }
]

llm_wrapper = LLMWrapper()

while True:
    user_message = input()
    messages.append({
        "role": "user",
        "content": user_message
    })
    response = llm_wrapper.open_ai_chat_complete(
        params={"messages": messages, "model": "gpt-3.5-turbo"}
    )
    message = dict(response["choices"][0]["message"])
    messages.append(message)
    print(message["content"])


