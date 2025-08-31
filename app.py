import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
st.title("LET'S FIND YOUR TMOCKOC EPISODE")

st.title("🎬 LET'S FIND YOUR TMKOC EPISODE 🎭")
st.write("Got a storyline in mind but can't recall the episode? 🤔")
st.write("Don't worry! Just describe the **main story** or **key events** of the episode (not dialogues).")


user_input=st.text_input("Write whatever you remembered in english")
vectorstore= FAISS.load_local("episodes_faiss", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
if user_input:
    results = retriever.get_relevant_documents(user_input)
    st.subheader("🔹 Top Episode Suggestions:")
    
    for i, doc in enumerate(results, start=1):
        st.markdown(f"### 🎥 Suggestion {i}")
        st.write(f"**Episode Number:** {doc.metadata.get('episode_number', 'N/A')}")
        st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
        st.write(f"**Description:** {doc.metadata.get('description', 'N/A')}")
        st.markdown("---")