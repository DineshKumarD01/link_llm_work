from llama_cpp import Llama

llm = Llama(
    model_path="models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    n_ctx=4096,
    n_threads=8,        # adjust based on CPU cores
    n_gpu_layers=35     # set >0 if GPU available
)

def build_prompt(query, chunks):

    context = "\n\n".join(chunks)

    prompt = f"""
You are a technical assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say that the context does not contain the answer.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt

def generate_answer(query, chunks):

    prompt = build_prompt(query, chunks)

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        top_p=0.9
    )

    return output["choices"][0]["text"]