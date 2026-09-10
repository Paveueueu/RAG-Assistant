import streamlit as st
import rag


state = st.session_state

if "paths" not in state:
    state.paths = []

if 'chat_history' not in st.session_state:
    state.chat_history = [("assistant", "Hello!", None)]

def add_path():
    if state.new_path:
        if state.new_path not in state.paths:
            state.paths.append(state.new_path)
            n = rag.index([state.new_path])
            st.toast(f"Indexed {n} chunks from {state.new_path}")
        state.new_path = ""

def remove_path(i):
    state.paths.pop(i)


st.set_page_config(page_title="RAG Assistant")



with st.container(border=True):
    with st.form(key="form", border=False):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.text_input(
                label="Path",
                label_visibility='hidden',
                placeholder="Add path to index", 
                key="new_path"
            )
            st.form_submit_button(
                "Add", 
                icon=":material/add:", 
                on_click=add_path
            )

    if state.paths:
        with st.container(gap=None, border=True):
            for i, path in enumerate(state.paths):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.container().write(path)
                    st.button(
                        ":material/delete:",
                        type="tertiary",
                        on_click=remove_path,
                        args=[i],
                    )
    else:
        st.info("No paths specified! Add some.")



for source, message, context in state.chat_history:
    with st.chat_message(source):
        st.write(message)
        if context:
            st.write(context)

prompt = st.chat_input("Ask me something...")
if prompt:
    state.chat_history.append(("user", prompt, None))
    with st.chat_message("user"):
        st.write(prompt)

    context, sources = rag.retrieve(prompt)
    with st.chat_message("assistant"):
        if context is None:
            answer = "No indexed data. Add paths that contain data to index."
            sources = None
            st.write(answer)
        else:
            answer = st.write_stream(rag.answer_stream(prompt, context))
            if sources:
                st.write(sources)

    state.chat_history.append(("assistant", answer, sources))