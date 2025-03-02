# Import necessary libraries
from sentence_transformers import SentenceTransformer
import json
import streamlit as st
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

load_dotenv()
# Set up Streamlit page
st.set_page_config(page_title="Patient Information QA System", layout="wide")
st.title("Ask EHR")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []


# Load the patient data
@st.cache_data
def load_patient_data():
    with open("data/final_Data.json", "r") as f:
        return json.load(f)


# Process and chunk the patient data using a flattened JSON approach with grouping
@st.cache_data
def process_patient_data(data):
    # First, flatten the JSON to get all paths
    def collect_paths(data, prefix="", paths=None):
        if paths is None:
            paths = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                collect_paths(value, new_prefix, paths)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                collect_paths(item, new_prefix, paths)
        else:
            # This is a leaf node
            paths.append((prefix, data))

        return paths

    # Collect all paths
    all_paths = collect_paths(data)

    # Group paths by their top-level key
    grouped_paths = {}
    for path, value in all_paths:
        top_level = path.split(".")[0]
        if top_level not in grouped_paths:
            grouped_paths[top_level] = []
        grouped_paths[top_level].append((path, value))

    # Create chunks from grouped paths
    chunks = []
    for top_level, paths in grouped_paths.items():
        # Format the section title
        section_title = top_level.replace("_", " ").title()
        content = f"## {section_title}\n\n"

        # Group by second level if available
        second_level_groups = {}
        for path, value in paths:
            parts = path.split(".")
            if len(parts) > 1:
                second_level = parts[1]
                if "[" in second_level:  # Handle array indices
                    second_level = second_level.split("[")[0]

                if second_level not in second_level_groups:
                    second_level_groups[second_level] = []
                second_level_groups[second_level].append((path, value))
            else:
                # Direct properties of the top level
                formatted_path = path.replace("_", " ").title()
                content += f"- **{formatted_path}**: {value}  \n"

        # Add second level groups
        for second_level, subpaths in second_level_groups.items():
            # Format the subsection title
            subsection_title = second_level.replace("_", " ").title()
            content += f"\n### {subsection_title}  \n"

            # Check if this is a list of items
            is_list = any("[" in path for path, _ in subpaths)

            if is_list:
                # Group by list index
                list_items = {}
                for path, value in subpaths:
                    # Extract the index from the path
                    match = re.search(r"\[(\d+)\]", path)
                    if match:
                        index = int(match.group(1))
                        if index not in list_items:
                            list_items[index] = []

                        # Get the property name (after the index)
                        remaining_path = path.split("]")[-1]
                        if remaining_path.startswith("."):
                            remaining_path = remaining_path[1:]

                        property_name = (
                            remaining_path.split(".")[-1] if remaining_path else "value"
                        )
                        list_items[index].append((property_name, value))

                # Format each list item
                for index, properties in sorted(list_items.items()):
                    # Try to find a name property for a better title
                    item_name = next(
                        (
                            value
                            for prop, value in properties
                            if prop.lower() in ["name", "condition", "type"]
                        ),
                        f"Item {index+1}",
                    )
                    content += f"\n#### {item_name}  \n"

                    for prop, value in properties:
                        formatted_prop = prop.replace("_", " ").title()
                        content += f"- **{formatted_prop}**: {value}  \n"

                    content += "\n"
            else:
                # Regular properties
                for path, value in subpaths:
                    # Get the property name (last part of the path)
                    property_name = path.split(".")[-1]
                    formatted_prop = property_name.replace("_", " ").title()
                    content += f"- **{formatted_prop}**: {value}  \n"

        chunks.append({"content": content, "source": section_title})

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


# Create FAISS index
@st.cache_resource
def create_faiss_index(embeddings):
    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize the vectors
    faiss.normalize_L2(embeddings)

    # Create the index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity

    # Add vectors to the index
    index.add(embeddings)
    return index


# Improve the function to find which chunks were actually used in the answer
def find_referenced_chunks(answer, chunks, relevant_chunks):
    """
    Identify which chunks were actually referenced in the answer with stricter filtering.
    """
    referenced_chunks = []
    seen_sources = set()  # Track sources we've already included
    answer_lower = answer.lower()

    # For each relevant chunk, check if specific content from it appears in the answer
    for chunk in relevant_chunks:
        # Extract specific data points from the chunk
        key_phrases = []

        # Extract values from markdown bullet points (more specific matching)
        bullet_points = re.findall(r"\*\*(.*?)\*\*: (.*?)(?:\n|$)", chunk)
        for key, value in bullet_points:
            # Only consider substantial values (not dates or single words)
            if len(value.strip()) > 5 and not re.match(
                r"^\d{4}-\d{2}-\d{2}$", value.strip()
            ):
                key_phrases.append((key.lower(), value.strip().lower()))

        # Check if any specific key-value pair is in the answer
        is_referenced = False
        matching_phrases = []

        for key, phrase in key_phrases:
            # Check for the specific value
            if phrase in answer_lower and len(phrase) > 5:
                is_referenced = True
                matching_phrases.append((key, phrase))

        # Only include chunks with specific matching content
        if is_referenced:
            # Get the source from the original chunk
            source = next(
                (c["source"] for c in chunks if c["content"] == chunk), "Unknown"
            )

            # Skip if we've already included this source
            if source in seen_sources:
                continue

            seen_sources.add(source)

            # Create a mini-chunk with just the matching information
            mini_chunk = ""

            # Get the section title from the chunk
            section_match = re.search(r"^## (.*?)$", chunk, re.MULTILINE)
            if section_match:
                section_title = section_match.group(1)
                mini_chunk += f"## {section_title}\n\n"

            # Add only the matching bullet points
            for key, phrase in matching_phrases:
                # Find the original formatting of the key and value
                for original_key, original_value in bullet_points:
                    if original_key.lower() == key and phrase in original_value.lower():
                        mini_chunk += f"- **{original_key}**: {original_value}\n"

            # Add this mini-chunk to the referenced chunks
            if mini_chunk:
                referenced_chunks.append(
                    {"content": mini_chunk.strip(), "source": source}
                )

    # If no chunks were referenced, include a minimal version of the most relevant chunk
    if not referenced_chunks and relevant_chunks:
        # Get the most relevant chunk
        most_relevant = relevant_chunks[0]
        source = next(
            (c["source"] for c in chunks if c["content"] == most_relevant), "Unknown"
        )

        # Extract the section title
        section_match = re.search(r"^## (.*?)$", most_relevant, re.MULTILINE)
        section_title = section_match.group(1) if section_match else "Information"

        # Create a minimal chunk with just the section title
        mini_chunk = f"## {section_title}\n\n(Most relevant section)"

        referenced_chunks.append({"content": mini_chunk, "source": source})

    return referenced_chunks


# Search for similar chunks
def search_similar_chunks(query, index, embeddings, embedding_model, chunks, k=2):
    # Create query embedding
    query_vector = embedding_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    # Detect if this is likely a list-type question
    list_keywords = ["list", "all", "what are", "medications", "medicines", "drugs"]
    is_list_question = any(keyword in query.lower() for keyword in list_keywords)

    # For list-type questions, increase k to get more comprehensive results
    if is_list_question:
        k = max(k, 4)  # Get at least 4 chunks for list questions

    # Search for similar chunks
    D, I = index.search(query_vector, k)

    # For list questions about a specific topic, try to find all related chunks
    if is_list_question:
        # Get the topic from the most relevant chunk
        most_relevant_source = chunks[I[0][0]]["source"]
        topic = (
            most_relevant_source.split(".")[0]
            if "." in most_relevant_source
            else most_relevant_source
        )

        # Find all chunks related to this topic
        related_indices = [
            i
            for i, chunk in enumerate(chunks)
            if topic.lower() in chunk["source"].lower()
        ]

        # Combine the vector-retrieved indices with topic-related indices
        all_indices = list(I[0]) + [idx for idx in related_indices if idx not in I[0]]

        # Limit to a reasonable number
        all_indices = all_indices[:6]

        # Return content from all these chunks
        return [chunks[i]["content"] for i in all_indices]

    # For regular questions, just return the vector-retrieved chunks
    return [chunks[i]["content"] for i in I[0]]


# Add a fallback response generator using the T5 model
def generate_fallback_response(query, context):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = (
        f"Answer this question based on the context: {query}\n\nContext: {context}"
    )
    input_ids = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).input_ids
    outputs = model.generate(
        input_ids, max_length=100, min_length=10, num_beams=1, early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Replace OpenAI client with HuggingFace client
@st.cache_resource
def get_inference_client():
    token = os.environ.get("HF_TOKEN")
    if not token:
        st.warning("⚠️ HuggingFace token not found. Using fallback model.")
        return None
    return InferenceClient(token=token)


def generate_response(client, query, context):
    try:
        # Create a more structured prompt that clearly defines the task
        prompt = f"""You are a medical assistant answering a single question about a patient based on their medical records.

Context from patient records:
{context}

Question: {query}

Provide a direct, concise answer to this specific question only. Do not generate additional questions or answers.

Answer:"""

        response = client.text_generation(
            prompt,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_new_tokens=500,
            temperature=0.1,  # Lower temperature for more focused responses
            do_sample=True,
            top_p=0.95,
            stop_sequences=[
                "Question:",
                "\n\n",
            ],  # Stop if it tries to generate a new question
        )

        # Clean up the response to remove any hallucinated Q&A pairs
        cleaned_response = response.split("Question:")[0].strip()
        cleaned_response = cleaned_response.split("\n\n")[0].strip()

        return cleaned_response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Main application
patient_data = load_patient_data()
chunks = process_patient_data(patient_data)
embedding_model = load_embedding_model()
embeddings = generate_embeddings(chunks, embedding_model)
index = create_faiss_index(embeddings)
inference_client = get_inference_client()

# Display patient overview
st.sidebar.header("Patient Overview")
st.sidebar.write(f"**ID:** {patient_data['patient']['id']}")
st.sidebar.write(f"**DOB:** {patient_data['patient']['dob']}")
st.sidebar.write(f"**NHS Number:** {patient_data['patient']['nhs_number']}")
st.sidebar.write(f"**Phone:** {patient_data['patient']['contact']['phone']}")
st.sidebar.write(f"**Address:** {patient_data['patient']['contact']['address']}")
st.sidebar.write(
    f"**GP:** {patient_data['patient']['gp']['name']}, {patient_data['patient']['gp']['practice']}"
)

# Add sidebar navigation links
st.sidebar.header("Other links")
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
    st.session_state.messages = []
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "references" in message:
            with st.expander("References"):
                st.text(message["references"])

# Chat input
if query := st.chat_input("Ask a question about the patient"):
    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                # Search for similar chunks
                relevant_chunks = search_similar_chunks(
                    query, index, embeddings, embedding_model, chunks
                )
                # Create context from relevant chunks
                context = "\n\n".join(relevant_chunks)

                # Generate answer using OpenAI if client is available, otherwise use fallback
                if inference_client:
                    try:
                        answer = generate_response(inference_client, query, context)
                    except Exception as e:
                        st.warning(f"API error: {str(e)}. Using fallback model.")
                        answer = generate_fallback_response(query, context)
                else:
                    answer = generate_fallback_response(query, context)

                # Find which chunks were actually referenced in the answer
                referenced_chunks = find_referenced_chunks(
                    answer, chunks, relevant_chunks
                )

                # Display answer
                st.write(answer)

                # Display only the referenced chunks
                with st.expander("References"):
                    if referenced_chunks:
                        for chunk in referenced_chunks:
                            st.markdown(f"**Source: {chunk['source']}**")
                            st.markdown(
                                chunk["content"]
                            )  # Use markdown to render the formatted chunk
                            st.markdown("---")
                    else:
                        st.write("No specific references found for this answer.")

                # Add assistant response to chat history with only referenced chunks
                references = "\n\n".join(
                    [
                        f"Source: {chunk['source']}\n{chunk['content']}"
                        for chunk in referenced_chunks
                    ]
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "references": references,
                    }
                )

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
# Display all chunks for inspection
with st.expander("View All Patient Data"):
    for i, chunk in enumerate(chunks):
        st.markdown(f"**Chunk {i+1}: {chunk['source']}**")
        st.text(chunk["content"])
        st.markdown("---")
