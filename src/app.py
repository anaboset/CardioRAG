from vectordb import VectorDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

class CV_RAG_Assistant:
    def __init__(self):
        self.db = VectorDB(collection_name="cv_pharmacotherapy")
        self.llm = self._get_llm()
        
        self.prompt = ChatPromptTemplate.from_template("""
You are a cardiologist consultant. Use only the provided context from the cardiovascular pharmacotherapy textbook.

Context:
{context}

Question: {question}

Instructions:
- Diagnose based on symptoms, risk factors, and guidelines.
- Recommend first-line drugs, doses, and monitoring.
- Cite the source chunk (use chunk_id).
- If uncertain, say "Not enough information in the source."

Answer:
""")
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _get_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            return ChatOpenAI(model="gpt-4o", temperature=0.0)
        elif os.getenv("GROQ_API_KEY"):
            return ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)
        else:
            raise ValueError("Set OPENAI_API_KEY or GROQ_API_KEY")

    def ask(self, question: str, n_results: int = 5) -> str:
        results = self.db.search(question, n_results)
        context = "\n\n".join([
            f"[Chunk {m['chunk_id']}] {doc}"
            for doc, m in zip(results["documents"], results["metadatas"])
        ])
        return self.chain.invoke({"context": context, "question": question})

# === Interactive Mode ===
if __name__ == "__main__":
    assistant = CV_RAG_Assistant()
    print("Cardiovascular RAG Assistant Ready!")
    print("Ask about diagnosis, drugs, guidelines... (type 'quit' to exit)\n")
    
    while True:
        q = input("Q: ").strip()
        if q.lower() == "quit":
            break
        if q:
            print("A:", assistant.ask(q))
            print("-" * 60)