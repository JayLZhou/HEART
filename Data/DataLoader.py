"""
Responsible for data loading.
"""

import json
import typing as T
from pathlib import Path
from llama_index.core import Document

from Schema.QASchema import QAPair
from Common.Logger import logger

class SimpleDataset:
    """Pure dataset interface - no configuration dependencies"""
    
    def __init__(self, corpus_file: str, qa_file: str, dataset_name: str):
        self.corpus_file = Path(corpus_file)
        self.qa_file = Path(qa_file) 
        self.dataset_name = dataset_name
        
        # Validate file existence
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        if not self.qa_file.exists():
            raise FileNotFoundError(f"QA file not found: {qa_file}")
            
        logger.info(f"Dataset initialized: {dataset_name}")
        logger.info(f"  Corpus: {corpus_file}")
        logger.info(f"  QA pairs: {qa_file}")
    
    def load_qa_pairs(self) -> T.List[QAPair]:
        """Load QA pairs - supports JSON and JSONL formats, handles text_ground_truth field"""
        qa_pairs = []
        
        # First load corpus data and create ID to text mapping
        corpus_mapping = self._load_corpus_mapping()
        logger.info(f"Corpus mapping loaded, total {len(corpus_mapping)} documents")
        
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to determine file format
        if content.startswith('[') and content.endswith(']'):
            # Standard JSON array format
            logger.info(f"Detected standard JSON array format")
            qa_data = json.loads(content)
        else:
            # JSONL format (one JSON object per line)
            logger.info(f"Detected JSONL format, parsing line by line")
            qa_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        qa_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse failed at line {line_num}: {e}")
                        continue
                        
        logger.info(f"Successfully loaded {len(qa_data)} QA entries")
        
        # Convert to QAPair objects
        for i, item in enumerate(qa_data):
            # Handle unified field mapping
            question = item.get('query', item.get('question', ''))  # Support both query and question fields
            answer = item.get('answer_ground_truth', item.get('answer', ''))  # Support multiple answer field names
            
            # Handle supporting_facts - from metadata or direct field
            supporting_facts = []
            if 'metadata' in item and 'supporting_facts' in item['metadata']:
                supporting_facts = item['metadata']['supporting_facts']
            else:
                supporting_facts = item.get('supporting_facts', [])
            
            # Format context
            context_formatted = []
            if isinstance(supporting_facts, list) and supporting_facts:
                for fact in supporting_facts:
                    if isinstance(fact, list) and len(fact) >= 2:
                        context_formatted.append({"entity": fact[0]})
                    elif isinstance(fact, str):
                        context_formatted.append({"entity": fact})
            
            # Handle difficulty and type
            difficulty = 'unknown'
            qtype = 'multihop'
            if 'metadata' in item:
                metadata = item['metadata']
                difficulty = metadata.get('difficulty', 'unknown')
                qtype = metadata.get('type', 'multihop')
            
            # Handle text_ground_truth field - map IDs to actual text content
            text_ground_truth_content = []
            text_ground_truth_ids = item.get('text_ground_truth', [])
            
            if text_ground_truth_ids and corpus_mapping:
                for doc_id in text_ground_truth_ids:
                    if str(doc_id) in corpus_mapping:
                        text_ground_truth_content.append(corpus_mapping[str(doc_id)])
                    else:
                        logger.warning(f"Document ID not found in corpus: {doc_id}")
            
            # Debug information
            if i < 3:  # Only output debug info for first 3 samples
                logger.info(f"Sample {i}: text_ground_truth_ids={text_ground_truth_ids}")
                logger.info(f"Sample {i}: text_ground_truth_content count={len(text_ground_truth_content)}")
            
            # Create QAPair object
            qa_pairs.append(QAPair(
                question=question,
                answer=answer,
                id=item.get('id', f"{self.dataset_name}_{i}"),
                context=context_formatted,
                supporting_facts=supporting_facts,
                difficulty=difficulty,
                qtype=qtype,
                dataset_name=self.dataset_name,
                text_ground_truth=text_ground_truth_content  # Add mapped text content
            ))
                
        return qa_pairs
    
    def load_corpus(self) -> T.List[Document]:
        """Load corpus documents - supports JSON and JSONL formats"""
        documents = []
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Try to determine file format
        if content.startswith('[') and content.endswith(']'):
            # Standard JSON array format
            logger.info(f"Detected standard JSON array format (corpus)")
            corpus_data = json.loads(content)
        elif content.startswith('{') and content.count('\n') > 0:
            # JSONL format (one JSON object per line)
            logger.info(f"Detected JSONL format (corpus), parsing line by line")
            corpus_data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        corpus_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Corpus line {line_num} JSON parse failed: {e}")
                        continue
        else:
            # Try to parse as dictionary format
            try:
                corpus_data = json.loads(content)
                logger.info(f"Detected dictionary format (corpus)")
            except json.JSONDecodeError:
                logger.error(f"Unable to recognize corpus file format")
                return []
                        
        logger.info(f"Successfully loaded {len(corpus_data)} corpus entries")
        
        # Handle different data formats
        if isinstance(corpus_data, dict):
            # Format 1: {doc_id: text, ...}
            for doc_id, text in corpus_data.items():
                documents.append(Document(
                    text=text,
                    metadata={"id": doc_id, "dataset": self.dataset_name}
                ))
        elif isinstance(corpus_data, list):
            # Format 2: [{"id": "...", "text": "..."}, ...] or other list formats
            for i, doc in enumerate(corpus_data):
                if isinstance(doc, dict):
                    # Unified format handling
                    text = doc.get('text', str(doc))
                    doc_id = doc.get('id', doc.get('title', f'doc_{i}'))
                    title = doc.get('title', doc_id)
                    
                    documents.append(Document(
                        text=text,
                        metadata={
                            "title": title, 
                            "id": str(doc_id), 
                            "dataset": self.dataset_name
                        }
                    ))
                else:
                    # Simple string format
                    documents.append(Document(
                        text=str(doc),
                        metadata={"id": str(i), "dataset": self.dataset_name}
                    ))
        
        return documents
    
    def _load_corpus_mapping(self) -> T.Dict[str, str]:
        """Load corpus and create ID to text mapping"""
        corpus_mapping = {}
        
        if not self.corpus_file.exists():
            logger.warning(f"Corpus file does not exist: {self.corpus_file}")
            return corpus_mapping
            
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Try to determine file format
            if content.startswith('[') and content.endswith(']'):
                # Standard JSON array format
                corpus_data = json.loads(content)
            elif content.startswith('{') and content.count('\n') > 0:
                # JSONL format
                corpus_data = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            corpus_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                # Try to parse as dictionary format
                corpus_data = json.loads(content)
            
            # Create ID to text mapping
            if isinstance(corpus_data, list):
                for doc in corpus_data:
                    if isinstance(doc, dict) and 'id' in doc and 'text' in doc:
                        corpus_mapping[str(doc['id'])] = doc['text']
            elif isinstance(corpus_data, dict):
                # If dictionary format {id: text, ...}
                for doc_id, text in corpus_data.items():
                    corpus_mapping[str(doc_id)] = text
                    
        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            
        return corpus_mapping
    
    def iter_grounding_data(self, partition="test") -> T.Iterator[Document]:
        """Compatible with StudyConfig.dataset interface: returns an iterator of corpus documents"""
        documents = self.load_corpus()
        for doc in documents:
            yield doc
    
    def model_dump(self) -> T.Dict[str, T.Any]:
        """Compatible with Pydantic model interface: returns model dictionary"""
        return {
            "xname": "simple_dataset",
            "dataset_name": self.dataset_name,
            "corpus_file": str(self.corpus_file),
            "qa_file": str(self.qa_file),
            "partition_map": {"test": "test", "train": "train"},
            "subset": "default",
            "grounding_data_path": str(self.corpus_file)
        }

def create_simple_dataset(corpus_file: str, qa_file: str, dataset_name: str) -> SimpleDataset:
    """Create a simple dataset"""
    return SimpleDataset(corpus_file, qa_file, dataset_name)