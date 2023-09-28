import streamlit as st
import pandas as pd
import os
import openai
import nltk  # Add this import
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import convert_file_to_embeddings as embeddings_converter

# Add the following lines to download the 'punkt' resource
nltk.download('punkt')

# openai API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# App information and setup
project_title = "Chat with Custom Data"
project_desc = """
This demo allows users to chat with ChatGPT-4 and their own custom data. 
The data can be uploaded as a .pdf or .txt file, or can be copy-pasted onto the text area. 
The data will be used as context provided to ChatGPT-4."""

project_icon = "icon.png"
st.set_page_config(page_title=project_title, initial_sidebar_state='expanded', page_icon=project_icon)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Set default model
if "model" not in st.session_state:
    st.session_state.model = "gpt-4"

# Set other default session state variables
if "use_custom_data" not in st.session_state:
    st.session_state.use_custom_data = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "custom_data_text" not in st.session_state:
    st.session_state.custom_data_text = ""
if "file_embeddings" not in st.session_state:
    st.session_state.file_embeddings = None
if "text_embeddings" not in st.session_state:
    st.session_state.text_embeddings = None


def create_embeddings_context(question, embeddings_list, max_len=1800, size="ada"):
    """Create a context for a question by finding the most similar context from the list of embeddings"""
    # Convert question to embeddings
    q_embeddings = get_embedding(question, engine='text-embedding-ada-002')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for file_embeddings in embeddings_list:
        # Get the distances from the embeddings
        file_embeddings['distances'] = distances_from_embeddings(q_embeddings, file_embeddings['embeddings'].values,
                                                                 distance_metric='cosine')

        for i, row in file_embeddings.sort_values('distances', ascending=True).iterrows():
            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4
            # If the context is too long, break
            if cur_len > max_len:
                break
            # Else add it to the text that is being returned
            returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def convert_inputs_to_embeddings():
    """Converts uploaded files and text area input into embeddings"""
    st.session_state.file_embeddings = []  # Initialize a list to store embeddings for multiple files

    if st.session_state.uploaded_file is not None:
        for uploaded_file in st.session_state.uploaded_file:
            if uploaded_file.type == "text/plain":
                file_embeddings = embeddings_converter.generate_embeddings(openai.api_key, uploaded_file, is_text_file=True)
                st.session_state.file_embeddings.append(file_embeddings)
                print(f"Text File Embeddings ({uploaded_file.name}):")
                print(file_embeddings)  # Print the file embeddings
            elif uploaded_file.type == "application/pdf":
                file_embeddings = embeddings_converter.generate_embeddings(openai.api_key, uploaded_file, is_pdf_file=True)
                st.session_state.file_embeddings.append(file_embeddings)
                print(f"PDF File Embeddings ({uploaded_file.name}):")
                print(file_embeddings)  # Print the file embeddings

    if st.session_state.custom_data_text != "":
        st.session_state.text_embeddings = embeddings_converter.generate_embeddings(openai.api_key,
                                                                                    st.session_state.custom_data_text)
        print("Text Area Embeddings:")
        print(st.session_state.text_embeddings)  # Print the text area embeddings

def get_context():
    file_embeddings_context = ""
    text_embeddings_context = ""
    last_message = st.session_state.messages[-1]['content']

    if st.session_state.use_custom_data:
        if st.session_state.file_embeddings is not None:
            file_embeddings_context = create_embeddings_context(question=last_message,
                                                                embeddings_list=st.session_state.file_embeddings,
                                                                max_len=900)
        if st.session_state.text_embeddings is not None:
            text_embeddings_context = create_embeddings_context(question=last_message,
                                                                embeddings_list=st.session_state.text_embeddings,
                                                                max_len=900)

    context = f"""
Assume the persona of 'Custom GPT', a chatbot specialized in responding to questions about a user's custom data.
Your primary function is to converse with the user in a lively and friendly manner.
Prioritize using the provided [CONTEXT EMBEDDINGS] generated from the user's custom data.
Do not inform users about the use of [CONTEXT EMBEDDINGS] or your general knowledgebase.
Refrain from responding to queries that violate standard moderation policies. Users should be treated with respect and courtesy, and inappropriate content should not be addressed.
[CONTEXT EMBEDDINGS]\n{file_embeddings_context}\n{text_embeddings_context}
"""

    return context



def main():
    head_col = st.columns([1, 8])
    with head_col[0]:
        st.image(project_icon)
    with head_col[1]:
        st.title(project_title)
    st.markdown(project_desc)

    # Reset chat
    reset_chat = st.button("Reset Chat")
    if reset_chat:
        st.session_state.messages = []
        st.success("Chat has been reset.")
    st.markdown("***")
    #########################################
    with st.sidebar:
        st.session_state.model = st.selectbox("Select a model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        st.session_state.use_custom_data = st.checkbox("Chat with Custom Data")
        if st.session_state.use_custom_data:
            st.session_state.uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"],
                                                              accept_multiple_files=True)
            st.session_state.custom_data_text = st.text_area("Or copy-paste your data here", height=200)
            st.write(
                "**Note:** OpenAI API has a request limit; if your custom data is large, you may have to wait a while for the embeddings to be generated.")
            if st.button("Generate Embeddings"):
                with st.spinner("Generating embeddings..."):
                    convert_inputs_to_embeddings()
                st.success("Embeddings have been generated.")
            st.write(f"File embeddings generated? {st.session_state.file_embeddings is not None}")
            st.write(f"Text embeddings generated? {st.session_state.text_embeddings is not None}")
            if st.button("Reset Embeddings"):
                st.session_state.file_embeddings = None
                st.session_state.text_embeddings = None
                st.success("Embeddings have been reset.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(placeholder="Type a message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Build messages to be sent to ChatGPT
            context = get_context()
            messages = [{"role": "system", "content": context}]
            # Add the last 20 messages in the conversation to the messages list
            messages.extend(st.session_state.messages[-20:])
            # Generate response from ChatGPT
            for response in openai.ChatCompletion.create(
                    model=st.session_state.model,
                    messages=messages,
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
