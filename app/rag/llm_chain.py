"""
LLM Chain for Answer Generation

This module handles the final step of RAG:
- Takes retrieved document chunks
- Constructs a prompt with context
- Calls OpenAI's GPT-4 to generate an answer

The LLM is instructed to:
- Only answer based on provided context
- Admit when it doesn't know
- Include citations when possible
"""

from openai import OpenAI
import os

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(question: str, chunks: list):
    """
    Generate an answer to a question using retrieved context chunks.
    
    This implements the "Generation" step of RAG (Retrieval-Augmented Generation).
    The LLM is given relevant document chunks as context and asked to answer
    the question based only on that context.
    
    Args:
        question (str): The user's question
        chunks (list): List of relevant document chunks from vector search
                      Each chunk is a dict with 'content' and 'metadata'
    
    Returns:
        str: The generated answer from GPT-4
        
    Example:
        chunks = [
            {"content": "Paris is the capital of France.", "metadata": {...}},
            {"content": "The Eiffel Tower is in Paris.", "metadata": {...}}
        ]
        answer = generate_answer("What is the capital of France?", chunks)
        # Returns: "The capital of France is Paris."
    """
    
    # Step 1: Combine all chunk contents into a single context string
    # Each chunk is separated by two newlines for readability
    context = "\n\n".join([c['content'] for c in chunks])
    
    # Step 2: Construct the prompt with instructions and context
    prompt = f"""
    You are an assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don't know.
    
    Context:
    {context}

    Question: {question}

    Answer concisely, and include citations if possible.
    """

    # Step 3: Call OpenAI API to generate the answer
    # Using gpt-4o model for high-quality responses
    # Temperature=0 makes responses more deterministic/consistent
    response = client.chat.completions.create(
        model="gpt-4o",  # GPT-4 Optimized model
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # 0 = deterministic, 2 = creative
    )
    
    # Step 4: Extract and return the text content from the response
    return response.choices[0].message.content
