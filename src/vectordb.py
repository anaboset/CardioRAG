# src/vectordb.py
import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Optional: LangChain splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_AVAILABLE = True
except Exception:
    _LANGCHAIN_AVAILABLE = False

# Optional: NLTK for sentences
try:
    import nltk
    nltk.download('punkt', quiet=True)
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


class VectorDB:
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        self.embedding_model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(path="./chroma_db")
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )
        print(f"Vector DB ready: {self.collection_name}")

    # ================================
    # 1. CHUNK TEXT (Advanced)
    # ================================
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        strategy: str = "recursive",
        overlap: int = 100,
        min_chunk_size: int = 100
    ) -> List[str]:
        if not text.strip():
            return []

        if strategy == "simple":
            return self._chunk_simple(text, chunk_size, overlap)
        elif strategy == "recursive":
            if not _LANGCHAIN_AVAILABLE:
                print("LangChain not available → using simple")
                return self._chunk_simple(text, chunk_size, overlap)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
            )
            chunks = splitter.split_text(text)
            return [c for c in chunks if len(c) >= min_chunk_size]
        elif strategy == "semantic":
            if not _NLTK_AVAILABLE:
                return self.chunk_text(text, chunk_size, "recursive", overlap)
            return self._chunk_semantic(text, chunk_size, overlap, min_chunk_size)
        else:
            raise ValueError("strategy must be 'simple', 'recursive', or 'semantic'")

    def _chunk_simple(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        cur = []
        cur_len = 0
        for word in words:
            word_len = len(word) + 1
            if cur_len + word_len > chunk_size and cur:
                chunks.append(" ".join(cur))
                cur = cur[-(overlap // 6):]
                cur_len = sum(len(w) + 1 for w in cur)
            cur.append(word)
            cur_len += word_len
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    def _chunk_semantic(self, text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> List[str]:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text] if len(text) >= min_chunk_size else []

        embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        embeddings = np.array(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=0.25,
        )
        labels = clustering.fit_predict(embeddings)

        clusters = {}
        for s, l in zip(sentences, labels):
            clusters.setdefault(l, []).append(s)

        final_chunks = []
        current = []
        current_len = 0

        for label in sorted(clusters):
            cluster_text = " ".join(clusters[label])
            if current and current_len + len(cluster_text) > chunk_size:
                chunk = " ".join(current)
                if len(chunk) >= min_chunk_size:
                    final_chunks.append(chunk)
                overlap_sents = self._take_last_sentences(current, overlap)
                current = overlap_sents + clusters[label]
                current_len = sum(len(s) for s in current) + len(current)
            else:
                current.extend(clusters[label])
                current_len += len(cluster_text) + len(clusters[label])

        if current:
            chunk = " ".join(current)
            if len(chunk) >= min_chunk_size:
                final_chunks.append(chunk)
        return final_chunks

    @staticmethod
    def _take_last_sentences(sentences: List[str], overlap_chars: int) -> List[str]:
        kept = []
        used = 0
        for s in reversed(sentences):
            if used + len(s) > overlap_chars:
                break
            kept.append(s)
            used += len(s) + 1
        return list(reversed(kept))

    # ================================
    # 2. ADD DOCUMENTS
    # ================================
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        print(f"Adding {len(documents)} documents...")
        all_chunks = []
        all_ids = []
        all_metadatas = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(content, strategy="recursive")  # or "semantic"

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append({**metadata, "chunk_id": chunk_idx})

        if all_chunks:
            embeddings = self.embedding_model.encode(all_chunks).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
        print("Documents added!")

    # ================================
    # 3. SEARCH
    # ================================
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        query_emb = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }