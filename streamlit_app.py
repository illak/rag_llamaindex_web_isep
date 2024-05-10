import streamlit as st
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


def load_data_from_url(url, api_key):
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        [url]
    )

    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2000, 1000, 500])
    nodes = node_parser.get_nodes_from_documents(documents)

    leaf_nodes = get_leaf_nodes(nodes)
    #root_nodes = get_root_nodes(nodes)

    # define storage context
    docstore = SimpleDocumentStore()

    # insert nodes into docstore
    docstore.add_documents(nodes)

    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore, )

    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(
        model_name=model_name, api_key=api_key, title="this is a document"
    )

    ## Load index into vector index
    base_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )


    base_retriever = base_index.as_retriever(similarity_top_k=6,)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

    return retriever


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]



def user_input(user_question, retriever, api_key):
    llm = Gemini(api_key=api_key)

    qa_prompt_str  = (
        "Contexto:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n\n"
        "Usted es un ayudante y debe responder a la pregunta del usuario usando el contexto provisto.\n"
        "Responda  de forma amable, detallada y extensa.\n"
        "Además responda con enlaces a los recursos en caso de que sea posible.\n"
        """Si no encuentra la respuesta en el contexto responda: "No pude encontrar información en el contexto provisto".\n\n"""
        "Pregunta: {query_str}\n"
    )

    # Text QA Prompt
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "No responda si el contexto no contiene información suficiente."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt_str ),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

    query_engine = RetrieverQueryEngine.from_args(retriever,
                                              llm=llm,
                                              text_qa_template=text_qa_template, 
                                              verbose=True)
    
    response = query_engine.query(user_question)

    return response



def main():

    st.set_page_config(
        page_title="Website chatbot",
        page_icon="🤖"
    )

  

    st.title("Website chatbot (Gemini 🤖  + LlamaIndex 🦙)")
    st.subheader("Chateando con el contenido de páginas web")
    
    # Ask for user input
    url = st.text_input("Ingrese una URL:", "")
    api_key = st.text_input("Ingrese la clave API de google Gemini:", "")
    
    if st.button("Listo"):
        # Perform actions using the URL and API key
        if url and api_key:
            # Your API integration code goes here
            # Cargamos los datos de la url
            with st.spinner("Procesando URL..."):
                st.session_state.retriever = load_data_from_url(url, api_key)
                st.success("Todo OK!")
        else:
            st.error("Ocurrió un error al cargar los datos!")


    st.write("Bienvenido/a al chat!")
    st.sidebar.button('Limpiar historial de chat', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Agregue la URL de una página web y haga preguntas"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):

                response = user_input(prompt, 
                                      st.session_state.retriever, 
                                      api_key)
                
                placeholder = st.empty()
                #full_response = ''
                #for item in response['output_text']:
                #    full_response += item
                #    placeholder.markdown(full_response)
                placeholder.markdown(str(response))

        if response is not None:
            message = {"role": "assistant", "content": str(response)}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()