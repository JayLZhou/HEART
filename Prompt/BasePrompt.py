"""
The prompt style is based on the LlamaIndex
Reference:
 - Prompts are from [LlamaIndex](https://github.com/jerryjliu/llama_index)
"""

from llama_index.core import PromptTemplate

# Default text QA prompt
DEFAULT_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: 
"""

# Concise prompt - emphasizes brief, direct answers
CONCISE_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
You are a helpful assistant. Answer the provided question based on the context information. Be concise!
Query: {query_str}
Answer:
"""

# Chain of Thought (CoT) prompt - encourages step-by-step reasoning
COT_TEXT_QA_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
Answer the provided question step-by-step based on the context information. Show your reasoning process.
Query: {query_str}
Answer:
"""

# Advanced RAG QA prompt with explicit thought process
RAG_QA_TEXT_PROMPT = """
Context information is below.
---------------------
{context_str}
---------------------
As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously.
Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions.
Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
Query: {query_str}
Answer:
"""

# Create PromptTemplate objects
qa_prompt = PromptTemplate(DEFAULT_TEXT_QA_PROMPT)
concise_prompt = PromptTemplate(CONCISE_TEXT_QA_PROMPT)
cot_prompt = PromptTemplate(COT_TEXT_QA_PROMPT)
rag_qa_prompt = PromptTemplate(RAG_QA_TEXT_PROMPT)