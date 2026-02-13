from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query, context, history):
    system_prompt = f"""
You are a grounded enterprise assistant.
Answer ONLY using the provided context.
If answer not found, say "Not found in provided documents."

Context:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content
