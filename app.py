"""Streamlit UI for the Bangla Literature Q&A Bot.

Usage:
    streamlit run app.py
"""

import streamlit as st
from generation import generate_answer

st.title("Bangla Literature Q&A Bot")

# Initialize session state
if "short_term_memory" not in st.session_state:
    st.session_state.short_term_memory = []

user_query = st.text_input("Enter your question here:")

if user_query:
    with st.spinner("Generating answer..."):
        try:
            answer = generate_answer(user_query, st.session_state.short_term_memory)
            st.markdown("**Answer:**")
            st.write(answer)
        except FileNotFoundError:
            st.error("FAISS index not found. Please run `python indexing.py` first.")
        except Exception as e:
            st.error(f"Error: {e}")

# Sidebar for chat history
st.sidebar.title("Previous Chat History")
if st.session_state.short_term_memory:
    for chat in reversed(st.session_state.short_term_memory):
        st.sidebar.markdown(chat)
else:
    st.sidebar.write("No previous chats yet.")
