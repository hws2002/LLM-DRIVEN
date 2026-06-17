import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from typing import List

logger = logging.getLogger("microscope")

def chunk_document(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int
) -> List[Document]:
    """English documentation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # English comment.
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    return chunks