from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(question: str, chunks: list):
    # 1. Prepare the context from your retrieved chunks
    context = "\n\n".join([c['content'] for c in chunks])
    
    prompt = f"""
    You are an assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know.
    
    Context:
    {context}

    Question: {question}

    Answer concisely, and include citations if possible.
    """

    # 2. Use the new client.chat.completions syntax
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=1.2
    )
    
    # 3. Access content using dot notation
    return response.choices[0].message.content