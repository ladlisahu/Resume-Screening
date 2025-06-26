import os
import re
import torch
import fitz
import tempfile
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 

# ChromaDB Setup
client = PersistentClient(path=".chromadb")
collection = client.get_or_create_collection("resumes")

# Embedding Model 
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def embed_texts(texts):
    return embedder.encode(texts).tolist()

# Load Local LLM 
@st.cache_resource
def load_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            use_auth_token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["jd", "resumes"],
    template="""
You are a hiring assistant. Given the job description and a list of candidate resumes, identify the top 5 most relevant candidates. For each, explain:
- Why this candidate is a good fit
- What skills or experiences match the job

Job Description:
{jd}

Candidate Resumes:
{resumes}

Output as a ranked list with justification.
"""
)
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Resume Parser 
def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def parse_resume(file_path):
    doc = fitz.open(file_path)
    text = " ".join([page.get_text() for page in doc])
    return clean_text(text)

# Threaded parsing
def parse_all_resumes(files):
    paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            paths.append(tmp.name)

    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(parse_resume, paths))
    return texts

# Chunking & Indexing 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def index_resumes_once(resume_texts):
    existing_ids = set(collection.get(include=[])["ids"])
    for i, text in enumerate(resume_texts):
        chunks = text_splitter.split_text(text)
        chunk_embeddings = embed_texts(chunks)
        for j, chunk in enumerate(chunks):
            uid = f"resume_{i}_{j}"
            if uid not in existing_ids:
                collection.add(
                    documents=[chunk],
                    embeddings=[chunk_embeddings[j]],
                    ids=[uid],
                    metadatas=[{"name": f"Candidate {i+1}"}]
                )

# Semantic Search
def get_top_k_resume_chunks(jd_text, k=10):
    query_embedding = embed_texts([jd_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results

# RAG Output 
def generate_rag_justification(jd_text, top_chunks):
    resume_block = ""
    added_candidates = set()
    for i in range(len(top_chunks["documents"][0])):
        name = top_chunks["metadatas"][0][i]["name"]
        if name not in added_candidates:
            resume = top_chunks["documents"][0][i]
            resume_block += f"\n\n{name}:\n{resume}"
            added_candidates.add(name)
    return llm_chain.run(jd=jd_text, resumes=resume_block)


# streamlit UI
st.set_page_config(page_title="Resume RAG Screener", layout="wide")
st.title("Resume Screening)")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
jd_text = st.text_area("Paste Job Description", height=250)

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.button("Screen Resumes"):
    if not uploaded_files or not jd_text.strip():
        st.warning("Please upload resumes and provide a job description.")
    else:
        st.info("Processing resumes...")

        resume_texts = parse_all_resumes(uploaded_files)

        if not st.session_state.indexed:
            index_resumes_once(resume_texts)
            st.session_state.indexed = True

        top_chunks = get_top_k_resume_chunks(clean_text(jd_text), k=10)
        result = generate_rag_justification(jd_text, top_chunks)

        st.success("Top Candidates Identified")
        st.markdown(result)
