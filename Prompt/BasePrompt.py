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

qa_prompt = PromptTemplate(DEFAULT_TEXT_QA_PROMPT)