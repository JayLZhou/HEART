"""
Prompt module for managing different prompt templates.
"""

from Prompt.BasePrompt import (
    qa_prompt,
    concise_prompt,
    cot_prompt,
    rag_qa_prompt,
)

__all__ = [
    'get_template',
    'qa_prompt',
    'concise_prompt',
    'cot_prompt',
    'rag_qa_prompt',

]

# template map
_TEMPLATE_MAP = {
    'default': qa_prompt,
    'concise': concise_prompt,
    'cot': cot_prompt,
    'rag_qa': rag_qa_prompt,
}


def get_template(template_name: str):
   
    template_name = template_name.lower()
    
    if template_name not in _TEMPLATE_MAP:
        available_templates = ', '.join(_TEMPLATE_MAP.keys())
        raise ValueError(
            f"Unknown template name: '{template_name}'. "
            f"Available templates: {available_templates}"
        )
    
    return _TEMPLATE_MAP[template_name]
