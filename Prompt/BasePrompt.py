"""
The prompt style is based on the LlamaIndex
Reference:
 - Prompts are from [LlamaIndex](https://github.com/jerryjliu/llama_index)
"""

# Default text QA prompt
DEFAULT_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Conclude with "Answer: 
Query: {query}
Answer: 
"""

# Concise prompt - emphasizes brief, direct answers
CONCISE_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
You are a helpful assistant. Answer the provided question based on the context information. Be concise!
Conclude with "Answer: 
Query: {query}
Answer:
"""

# Chain of Thought (CoT) prompt - encourages step-by-step reasoning
COT_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
Answer the provided question step-by-step based on the context information. Show your reasoning process.
Conclude with "Answer: 
Query: {query}
Answer:
"""

# Advanced RAG QA prompt with explicit thought process
RAG_QA_TEXT_PROMPT = """
Context information is below.
---------------------
{context}
---------------------
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously.
Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions.
Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
Query: {query}
Answer:
"""

# Create PromptTemplate objects
qa_prompt = DEFAULT_TEXT_QA_PROMPT
concise_prompt = CONCISE_TEXT_QA_PROMPT
cot_prompt = COT_TEXT_QA_PROMPT
rag_qa_prompt = RAG_QA_TEXT_PROMPT