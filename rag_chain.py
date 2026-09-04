"""
rag_chain.py

Core RAG logic pulled out of medibot.py so it can be imported by
BOTH the Streamlit app (medibot.py) and the FastAPI server (app.py),
instead of being duplicated or locked inside a Streamlit callback.

Nothing here imports streamlit or fastapi — this module knows nothing
about how it's being served.
"""

import os
import re
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Give a thorough, well-explained answer using everything relevant in the
context above — aim for several sentences or short paragraphs rather than
a one-liner, covering causes, symptoms, or next steps where the context
supports it. Write in plain sentences only — no markdown, no asterisks,
no bullet points, no headings. Start the answer directly. No small talk please.
"""


def plain_text_clean(text: str) -> str:
    """Strips markdown formatting the model may add despite the prompt
    asking it not to, and normalizes whitespace into plain paragraphs.
    Runs on every answer regardless of what the model actually outputs,
    so the API's output is guaranteed plain text either way."""
    # Bold/italic markers: **text**, __text__, *text*, _text_
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", text)
    # Markdown headings (#, ##, ...) and bullet/list markers at line starts
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^[\-\*\u2022]\s+", "", text)
    text = re.sub(r"(?m)^\d+\.\s+", "", text)
    # Collapse 3+ newlines down to a double newline (paragraph break),
    # and any run of spaces/tabs down to one space
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def set_custom_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(template=template, input_variables=["context", "question"])


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """Loaded once per process and cached — FAISS load + embedding model
    init is slow, so we don't want it happening on every request."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


@lru_cache(maxsize=1)
def get_qa_chain() -> RetrievalQA:
    """Builds (once) and returns the retrieval-QA chain.

    NOTE: this currently uses Groq (ChatGroq), matching medibot.py.
    The synopsis (section 8) specifies Google Gemini API instead — if
    you switch, swap this block for `langchain_google_genai.ChatGoogleGenerativeAI`
    and update the .env var name accordingly. Everything else in this
    file (retriever, prompt, chunking) stays the same either way.
    """
    vectorstore = get_vectorstore()

    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
        max_tokens=4096,  # generous budget: gpt-oss models use some of this on hidden
        # reasoning by default, so this leaves plenty left over for a full visible answer
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # more context chunks to draw on
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    )


def answer_query(question: str) -> dict:
    """The one function both Streamlit and FastAPI call.

    Returns: {"result": str, "sources": [ {"source": str, "page": int}, ... ]}
    """
    qa_chain = get_qa_chain()
    response = qa_chain.invoke({"query": question})

    sources = []
    for doc in response.get("source_documents", []):
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        })

    return {"result": plain_text_clean(response["result"]), "sources": sources}