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
# Map step: summarize a single chunk
MAP_TEXT_QA_PROMPT = """
Given the following context passage, briefly answer the question using only information from this passage.
If the passage is not relevant, respond with 'Not relevant'.
Question: {query}
Context: {context}
Brief answer:
"""

# Reduce step: synthesize from partial answers
REDUCE_TEXT_QA_PROMPT = """
Based on the following partial answers to the question, write a final comprehensive answer.
If the partial answers are unhelpful, say you do not know.
Question: {query}
Partial answers:
{context}
Final answer:
"""

# Refine step: iteratively refine existing answer with new context
REFINE_TEXT_QA_PROMPT = """
You have an existing answer to the question.
If the new context is useful, refine the answer. Otherwise keep the existing answer unchanged.
Question: {query}
Existing answer: {existing_answer}
New context: {context}
Refined answer:
"""

map_prompt = MAP_TEXT_QA_PROMPT
reduce_prompt = REDUCE_TEXT_QA_PROMPT
refine_prompt = REFINE_TEXT_QA_PROMPT
