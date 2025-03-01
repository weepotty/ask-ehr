# Import necessary libraries
from transformers import pipeline
import json
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit page
st.set_page_config(page_title="Patient Information QA System", layout="wide")
st.title("Ask EHR")

# Initialize chat history in session state if it doesn't exist
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# Load the patient data
@st.cache_data
def load_patient_data():
    with open("patient.json", "r") as f:
        return json.load(f)


# Process and chunk the patient data
@st.cache_data
def process_patient_data(patient_data):
    chunks = []

    # Process basic information
    basic_info = f"Patient name: {patient_data['name']}\n"
    basic_info += f"Date of birth: {patient_data['date_of_birth']}\n"
    basic_info += f"Gender: {patient_data['gender']}\n"
    chunks.append({"content": basic_info, "source": "Basic Information"})

    # Process contact information
    contact_info = "Contact Information:\n"
    for key, value in patient_data["contact_information"].items():
        contact_info += f"{key}: {value}\n"
    chunks.append({"content": contact_info, "source": "Contact Information"})

    # Process allergies
    allergies_info = "Allergies:\n"
    for allergy in patient_data["allergies"]:
        allergies_info += f"- {allergy['substance']}: {allergy['reaction']} (Severity: {allergy['severity']})\n"
    chunks.append({"content": allergies_info, "source": "Allergies"})

    # Process medical history
    for condition in patient_data["past_medical_history"]:
        condition_info = f"Medical Condition: {condition['condition']}\n"
        condition_info += f"Diagnosis date: {condition['diagnosis_date']}\n"
        condition_info += f"Status: {condition['status']}\n"
        condition_info += f"Medications: {', '.join(condition['medications'])}\n"
        chunks.append(
            {
                "content": condition_info,
                "source": f"Medical History - {condition['condition']}",
            }
        )

    # Process clinic letters
    for letter in patient_data["clinic_letters"]:
        letter_info = f"Clinic Letter ({letter['date']}):\n"
        letter_info += f"Specialty: {letter['specialty']}\n"
        letter_info += f"Consultant: {letter['consultant']}\n"
        letter_info += f"Summary: {letter['summary']}\n"
        chunks.append(
            {"content": letter_info, "source": f"Clinic Letter - {letter['date']}"}
        )

    # Process discharge summaries
    for summary in patient_data["discharge_summaries"]:
        summary_info = f"Discharge Summary ({summary['date']}):\n"
        summary_info += f"Hospital: {summary['hospital']}\n"
        summary_info += f"Reason for admission: {summary['reason_for_admission']}\n"
        summary_info += f"Summary: {summary['summary']}\n"
        chunks.append(
            {
                "content": summary_info,
                "source": f"Discharge Summary - {summary['date']}",
            }
        )

    # Process lab results
    for lab in patient_data["labs"]:
        lab_info = f"Lab Test ({lab['date']}):\n"
        lab_info += f"Test: {lab['test']}\n"
        if "result" in lab:
            lab_info += f"Result: {lab['result']}\n"
        elif "results" in lab:
            lab_info += "Results:\n"
            for test, result in lab["results"].items():
                lab_info += f"- {test}: {result}\n"
        lab_info += f"Reference range: {lab['reference_range']}\n"
        lab_info += f"Interpretation: {lab['interpretation']}\n"
        chunks.append(
            {
                "content": lab_info,
                "source": f"Lab Results - {lab['date']} - {lab['test']}",
            }
        )

    return chunks


# Generate embeddings for chunks
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def generate_embeddings(chunks, _model):
    texts = [chunk["content"] for chunk in chunks]
    embeddings = _model.encode(texts)
    return embeddings


# Load QA model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/bert-base-cased-squad2")


# Main application
patient_data = load_patient_data()
chunks = process_patient_data(patient_data)
embedding_model = load_embedding_model()
embeddings = generate_embeddings(chunks, embedding_model)
qa_model = load_qa_model()

# Display patient overview
st.sidebar.header("Patient Overview")
st.sidebar.write(f"**Name:** {patient_data['name']}")
st.sidebar.write(f"**DOB:** {patient_data['date_of_birth']}")
st.sidebar.write(f"**Gender:** {patient_data['gender']}")

# Add sidebar navigation links
st.sidebar.header("Navigation")
sidebar_options = [
    "Doctor's notes",
    "Nurse's notes",
    "Physiotherapy notes",
    "Discharge summaries",
    "Blood test results",
    "Imaging reports",
]

# Initialize selected option in session state if it doesn't exist
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

# Create clickable links for each option
for option in sidebar_options:
    st.sidebar.button(option)


# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.qa_history = []
    st.rerun()

# Display chat history
st.header("Chat History")
for i, qa_pair in enumerate(st.session_state.qa_history):
    st.markdown(f"**Question {i+1}:** {qa_pair['question']}")
    st.markdown(f"**Answer {i+1}:** {qa_pair['answer']}")
    st.markdown("---")


# Use a form to handle submission and clear the input
with st.form(key="query_form"):
    query = st.text_input("Your question:")
    submit_button = st.form_submit_button("Submit")

# Process the query when the form is submitted
if submit_button and query:
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query])[0]

    # Calculate similarity scores
    similarity_scores = cosine_similarity([query_embedding], embeddings)[0]

    # Get top 3 most relevant chunks
    top_indices = np.argsort(similarity_scores)[-3:][::-1]

    # Combine relevant contexts
    context = "\n\n".join([chunks[i]["content"] for i in top_indices])

    # Get answer from QA model
    with st.spinner("Generating answer..."):
        answer = qa_model(question=query, context=context)

    # Add to chat history
    st.session_state.qa_history.append(
        {
            "question": query,
            "answer": answer["answer"],
            "confidence": answer["score"],
            "sources": [
                {"source": chunks[i]["source"], "relevance": similarity_scores[i]}
                for i in top_indices
            ],
        }
    )

    # Display answer
    st.subheader("Answer")
    st.write(answer["answer"])
    st.write(f"Confidence: {answer['score']:.2%}")

    # Display source information
    st.subheader("Source Information")
    for i in top_indices:
        with st.expander(f"Source: {chunks[i]['source']}"):
            st.write(chunks[i]["content"])
            st.write(f"Relevance score: {similarity_scores[i]:.2%}")

# Display all chunks for inspection
with st.expander("View All Patient Data"):
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}: {chunk['source']}**")
        st.text(chunk["content"])
        st.markdown("---")
