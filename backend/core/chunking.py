import re
import spacy
import spacy.cli
import tiktoken

TOKEN_LIMIT = 8192  # OpenAI's embedding model limit
SUB_CHUNK_SIZE = 2000  # Safe sub-chunk size to avoid exceeding limits

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
# Load tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text):
    """Returns the token count of a given text."""
    return len(tokenizer.encode(text))

def split_chunk(chunk, max_tokens=SUB_CHUNK_SIZE):
    """Splits a chunk into smaller sub-chunks if it exceeds max_tokens."""
    tokens = tokenizer.encode(chunk)
    if len(tokens) <= max_tokens:
        return [chunk]  # Already within the limit

    sub_chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i + max_tokens]
        sub_chunks.append(tokenizer.decode(sub_tokens))
    
    return sub_chunks

def markdown_chunking(markdown_text, heading_level=2):
    pattern = rf'(?=^{"#" * heading_level} )'  # Regex to match specified heading level
    # chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    raw_chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)
    
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Ensure the chunk is within token limits
        chunks.extend(split_chunk(chunk, max_tokens=TOKEN_LIMIT // 2))  # Split large chunks
    
    return chunks
    
    # return [chunk.strip() for chunk in chunks if chunk.strip()]

def semantic_chunking(text, max_sentences=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
   
    chunks = []
    current_chunk = []
    for i, sent in enumerate(sentences):
        current_chunk.append(sent)
        if (i + 1) % max_sentences == 0:
            merged_chunk = " ".join(current_chunk)
            chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))  # Ensure token limits
            current_chunk = []
   
    if current_chunk:
        merged_chunk = " ".join(current_chunk)
        chunks.extend(split_chunk(merged_chunk, max_tokens=TOKEN_LIMIT // 2))
   
    return chunks

def sliding_window_chunking(text, chunk_size=SUB_CHUNK_SIZE, overlap=200):
    """Splits text into overlapping token-based chunks while ensuring token limits."""
    tokens = tokenizer.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        # Ensure the chunk doesn't exceed the max allowed token limit
        if count_tokens(chunk_text) > TOKEN_LIMIT // 2:
            chunks.extend(split_chunk(chunk_text, max_tokens=TOKEN_LIMIT // 2))
        else:
            chunks.append(chunk_text)

        if end == len(tokens):
            break  # Stop if we've reached the end
        
        start += chunk_size - overlap  # Slide the window with overlap
    
    return chunks