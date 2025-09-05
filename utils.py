from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import List
import os
import asyncio

FOLDER_PATH = "Docs"
load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# --- Helpers ---
def get_embeddings():
    """Ensure an asyncio loop exists before creating embeddings."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # no loop in this thread
        asyncio.set_event_loop(asyncio.new_event_loop())
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        collection_name="it_operations",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

# --- Document Loader ---
def _load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

def create_index_documents():
    """Load, split, and add docs into Chroma."""
    docs = _load_documents(FOLDER_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store = get_vector_store()
    vector_store.add_documents(documents=all_splits)


System_Prompt =(
    "You are an assistant for answering IT Operations questions"
    "Use the following pieces of retrieved context to answer the question."
    "Answer in a Detail way with markdown format."
    "If you don't know the answer, just say that you don't know."
    "Note: Please make sure to answer in a Markdown Format with Heading as title."
)

graph_builder = StateGraph(MessagesState)
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = get_vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for answering IT Operations questions"
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, just say that you don't know."
        "keep the answer Consise."
        "Note: Please make sure to answer in a Markdown Format with Heading as title."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    
    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

memory = MemorySaver()
graph_rag = graph_builder.compile(checkpointer=memory)