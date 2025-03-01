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
    with open("data/example.json", "r") as f:
        return json.load(f)


# Process and chunk the patient data
@st.cache_data
def process_patient_data(data):
    chunks = []
    patient_data = data["patient"]

    # Process basic information
    basic_info = f"Patient name: {patient_data['name']}\n"
    basic_info += f"Date of birth: {patient_data['dob']}\n"
    basic_info += f"NHS Number: {patient_data['nhs_number']}\n"
    basic_info += f"Address: {patient_data['address']}\n"
    basic_info += f"Phone: {patient_data['phone']}\n"
    chunks.append({"content": basic_info, "source": "Basic Information"})

    # Process GP information
    gp_info = "GP Information:\n"
    gp_info += f"Name: {patient_data['gp']['name']}\n"
    gp_info += f"Practice: {patient_data['gp']['practice']}\n"
    gp_info += f"Address: {patient_data['gp']['address']}\n"
    gp_info += f"Phone: {patient_data['gp']['phone']}\n"
    chunks.append({"content": gp_info, "source": "GP Information"})

    # Process diagnoses
    diagnoses_info = "Diagnoses:\n"
    for diagnosis in patient_data["diagnoses"]:
        diagnoses_info += f"- {diagnosis['condition']} (ICD-10: {diagnosis['icd_10_code']}), diagnosed on {diagnosis['date_diagnosed']}\n"
    chunks.append({"content": diagnoses_info, "source": "Diagnoses"})

    # Process allergies
    allergies_info = "Allergies:\n"
    for allergy in patient_data["allergies"]:
        allergies_info += f"- {allergy['substance']}: {allergy['reaction']} (Severity: {allergy['severity']})\n"
    chunks.append({"content": allergies_info, "source": "Allergies"})

    # Process medications
    medications_info = "Medications:\n"
    for medication in patient_data["medications"]:
        end_date = medication["end_date"] if medication["end_date"] else "ongoing"
        medications_info += f"- {medication['name']} {medication['dosage']} {medication['unit']} {medication['frequency']}, "
        medications_info += f"started {medication['start_date']}, {end_date}\n"
    chunks.append({"content": medications_info, "source": "Medications"})

    # Process seizure history
    seizure_info = "Seizure History:\n"
    for seizure in patient_data["seizure_history"]:
        seizure_info += f"- {seizure['date']}: {seizure['description']}\n"
    chunks.append({"content": seizure_info, "source": "Seizure History"})

    # Process social history
    social_info = "Social History:\n"
    social_info += f"Occupation: {patient_data['social_history']['occupation']}\n"
    social_info += (
        f"Smoking Status: {patient_data['social_history']['smoking_status']}\n"
    )
    social_info += f"Alcohol Consumption: {patient_data['social_history']['alcohol_consumption']}\n"
    social_info += (
        f"Living Situation: {patient_data['social_history']['living_situation']}\n"
    )
    chunks.append({"content": social_info, "source": "Social History"})

    # Process appointments
    for appointment in patient_data["appointments"]:
        appointment_info = f"Appointment ({appointment['date']}):\n"
        appointment_info += f"Type: {appointment['type']}\n"
        appointment_info += f"Location: {appointment['location']}\n"
        if "consultant" in appointment:
            appointment_info += f"Consultant: {appointment['consultant']}\n"
        elif "gp" in appointment:
            appointment_info += f"GP: {appointment['gp']}\n"
        appointment_info += f"Notes: {appointment['notes']}\n"
        chunks.append(
            {
                "content": appointment_info,
                "source": f"Appointment - {appointment['date']} - {appointment['type']}",
            }
        )

    # Process discharge summary
    discharge_info = (
        f"Discharge Summary ({patient_data['discharge_summary']['date']}):\n"
    )
    discharge_info += f"Admitting Diagnosis: {patient_data['discharge_summary']['admitting_diagnosis']}\n"
    discharge_info += f"Hospital: {patient_data['discharge_summary']['hospital']}\n"
    discharge_info += (
        f"Length of Stay: {patient_data['discharge_summary']['length_of_stay']}\n"
    )
    discharge_info += f"Summary: {patient_data['discharge_summary']['free_text']}\n"
    chunks.append(
        {
            "content": discharge_info,
            "source": f"Discharge Summary - {patient_data['discharge_summary']['date']}",
        }
    )

    # Process test results
    for test in patient_data["test_results"]:
        test_info = f"Test Result ({test['date']}):\n"
        test_info += f"Test: {test['test']}\n"
        test_info += f"Location: {test['location']}\n"
        if "findings" in test:
            test_info += f"Findings: {test['findings']}\n"
        elif "results" in test:
            test_info += "Results:\n"
            for result_name, result_data in test["results"].items():
                test_info += (
                    f"- {result_name}: {result_data['value']} {result_data['unit']} "
                )
                test_info += f"(Normal range: {result_data['normal_range']})\n"
        chunks.append(
            {
                "content": test_info,
                "source": f"Test Result - {test['date']} - {test['test']}",
            }
        )

    # Process theatre visits
    for visit in patient_data["theatre_visits"]:
        visit_info = f"Theatre Visit ({visit['date']}):\n"
        visit_info += f"Specialty: {visit['specialty']}\n"
        visit_info += (
            f"Procedure: {visit['procedure']} (Code: {visit['procedure_code']})\n"
        )
        visit_info += f"Admission ID: {visit['admission_id']}\n"
        visit_info += f"Procedure Time: {visit['procedure_start_time']} to {visit['procedure_end_time']}\n"
        chunks.append(
            {
                "content": visit_info,
                "source": f"Theatre Visit - {visit['date']} - {visit['procedure']}",
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
st.sidebar.write(f"**Name:** {patient_data['patient']['name']}")
st.sidebar.write(f"**DOB:** {patient_data['patient']['dob']}")
st.sidebar.write(f"**NHS Number:** {patient_data['patient']['nhs_number']}")

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


# Add an improved function to post-process answers for different question types
def process_answer(question, answer, context):
    # Convert question to lowercase for easier matching
    question_lower = question.lower()

    # Check for different question types
    if any(
        keyword in question_lower
        for keyword in ["medication", "medicine", "drug", "prescription"]
    ):
        if "Medications:" in context:
            # Extract medication information
            medications_section = context.split("Medications:")[1].split("\n\n")[0]
            medication_lines = [
                line.strip()
                for line in medications_section.split("\n")
                if line.strip().startswith("-")
            ]
            if medication_lines:
                medications = [
                    med.strip("- ").split(",")[0] for med in medication_lines
                ]
                return f"The patient is on the following medications: {', '.join(medications)}"

    elif any(keyword in question_lower for keyword in ["allergy", "allergic"]):
        if "Allergies:" in context:
            # Extract allergy information
            allergies_section = context.split("Allergies:")[1].split("\n\n")[0]
            allergy_lines = [
                line.strip()
                for line in allergies_section.split("\n")
                if line.strip().startswith("-")
            ]
            if allergy_lines:
                allergies = [
                    allergy.strip("- ").split(":")[0] for allergy in allergy_lines
                ]
                return (
                    f"The patient has the following allergies: {', '.join(allergies)}"
                )

    elif any(
        keyword in question_lower for keyword in ["diagnosis", "condition", "disease"]
    ):
        if "Diagnoses:" in context:
            # Extract diagnosis information
            diagnoses_section = context.split("Diagnoses:")[1].split("\n\n")[0]
            diagnosis_lines = [
                line.strip()
                for line in diagnoses_section.split("\n")
                if line.strip().startswith("-")
            ]
            if diagnosis_lines:
                diagnoses = [
                    diag.strip("- ").split(" (ICD")[0] for diag in diagnosis_lines
                ]
                return f"The patient has been diagnosed with: {', '.join(diagnoses)}"

    elif "test" in question_lower or "result" in question_lower:
        # Try to find test results in the context
        test_sections = []
        if "Test Result" in context:
            for section in context.split("Test Result"):
                if section.strip():
                    test_sections.append("Test Result" + section.split("\n\n")[0])

            if test_sections:
                return f"The patient has the following test results:\n" + "\n".join(
                    test_sections
                )

    # For other types of questions, try to improve the BERT answer
    bert_answer = answer["answer"]

    # If the answer is very short, try to find the sentence containing it for more context
    if len(bert_answer.split()) < 3:
        for sentence in context.split(". "):
            if bert_answer in sentence:
                return sentence + "."

    return bert_answer


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

    # Process the answer for special cases
    processed_answer = process_answer(query, answer, context)

    # Add to chat history
    st.session_state.qa_history.append(
        {
            "question": query,
            "answer": processed_answer,  # Use the processed answer
            "confidence": answer["score"],
            "sources": [
                {"source": chunks[i]["source"], "relevance": similarity_scores[i]}
                for i in top_indices
            ],
        }
    )

    # Display answer
    st.subheader("Answer")
    st.write(processed_answer)  # Display the processed answer
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
