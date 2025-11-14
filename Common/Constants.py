import os
from pathlib import Path

from loguru import logger
from enum import Enum
from typing import List
import typing as T
Process_tickers = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

DEFAULT_LLMS: T.List[str] = list(
    set(
        [
            "gpt-4o-mini",  # first LLM is the default
            "gpt-4o",
        ]
    )
)

PARAMETERS = [
    "rag_retriever",
    "reranker",
    "rag_mode",
    "sub_question_rag",
    "response_synthesizer_llm",
    "template_name",
]


Default_text_separator = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]

def get_package_root():

    package_root = Path.cwd()

    return package_root


def get_root():
    """Get the project root directory."""
    # Check if a project root is specified in the environment variable
    project_root_env = os.getenv("METAGPT_PROJECT_ROOT")
    if project_root_env:
        project_root = Path(project_root_env)
        logger.info(f"PROJECT_ROOT set from environment variable to {str(project_root)}")
    else:
        # Fallback to package root if no environment variable is set
        project_root = get_package_root()
      
    return project_root

PROJECT_ROOT = get_root()

CONFIG_ROOT = Path.home() / "Option"


# Timeout
USE_CONFIG_TIMEOUT = 0  # Using llm.timeout configuration.
LLM_API_TIMEOUT = 300

# Split tokens
GRAPH_FIELD_SEP = "<SEP>"

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

IGNORED_MESSAGE_ID = "0"



# Used for the Memory 

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_META_ROLE = "role"
MESSAGE_ROUTE_TO_ALL = "<all>"
MESSAGE_ROUTE_TO_NONE = "<none>"
NDIGITS = 4

# Used for Medical-Graph-RAG like

NODE_PATTERN = r"Node\(id='(.*?)', type='(.*?)'\)"
REL_PATTERN  = r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"

# Relationship(subj=Node(id=\'Scott Derrickson\', type=\'Person\'), obj=Node(id=\'Deliver Us From Evil\', type=\'Film\')

# For wiki-link
GCUBE_TOKEN = '07e1bd33-c0f5-41b0-979b-4c9a859eec3f-843339462'

hex_color = "#ea6eaf"
r = int(hex_color[1:3], 16)
g = int(hex_color[3:5], 16)
b = int(hex_color[5:7], 16)
ANSI_COLOR = f"\033[38;2;{r};{g};{b}m"
TOKEN_TO_CHAR_RATIO = 4


# Embedding models
DEFAULT_EMBEDDING_MODELS: T.List[str] = list(
    set(
        [
            "BAAI/bge-small-en-v1.5",  # first embedding model is the default
            "BAAI/bge-large-en-v1.5",
            "thenlper/gte-large",
            "mixedbread-ai/mxbai-embed-large-v1",
            "WhereIsAI/UAE-Large-V1",
            "avsolatorio/GIST-large-Embedding-v0",
            "w601sxs/b1ade-embed",
            "Labib11/MUG-B-1.6",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "FinLang/finance-embeddings-investopedia",
            "baconnier/Finance2_embedding_small_en-V1.5",
        ]
    )
)


TEMPLATE_NAMES = [
  "default", "concise", "cot", "rag_qa"
]



class TunerType(Enum):
    """Tuner type enumeration."""
    BO = "bo"  # Bayesian Optimization
    MAB = "mab"  # Multi-Armed Bandit
    OTHER = "other"  # Other optimization methods
    