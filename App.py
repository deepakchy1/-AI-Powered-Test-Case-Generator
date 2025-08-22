import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import fitz 
from PIL import Image
import re

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI-Powered Test Case Generator",
    page_icon="🧪"
)

# --- Helper Functions ---

def process_pdf(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        file_bytes = io.BytesIO(uploaded_file.read())
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

def parse_response(response_text):
    """
    Parses the AI's response to extract the summary and the markdown table.
    Returns a tuple: (summary_text, table_text)
    """
    # Use a regex to find the start of the markdown table
    match = re.search(r"\|\s*Test Case ID\s*\|", response_text)
    if match:
        table_start_index = match.start()
        summary = response_text[:table_start_index].strip()
        table_content = response_text[table_start_index:].strip()
        return summary, table_content
    return None, response_text


# --- Streamlit UI ---

# Header Section
st.title("🧪 AI-Powered Test Case Generator")
st.markdown("Upload a UI screenshot, a requirements PDF, or paste a user story to automatically generate comprehensive test cases.")

# Sidebar for API Key Input and Model Selection
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input(
        "Enter your Google AI API Key",
        type="password",
        help="Get your API key from Google AI Studio."
    )
    st.markdown("[Get your Google AI API key](https://makersuite.google.com/app/apikey)")
    
    st.header("🤖 AI Model Settings")
    model_name = st.selectbox(
        "Select an AI Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash"],
        index=0, # Default to gemini-1.5-flash
        help="Choose the model to generate the test cases. Gemini 1.5 Flash is recommended for its speed and performance."
    )


# --- Main Sequential Layout ---

# Step 1: Select Input Type
st.subheader("1. Select Input Type")
input_type = st.radio(
    "Choose the source of your requirements:",
    ("Image (UI Screenshot)", "PDF (Requirements Doc)", "Text (User Story)"),
    label_visibility="collapsed"
)

# Step 2: Provide Your Input & Context
st.subheader("2. Provide Your Input & Context")
uploaded_file = None
user_text = ""

# Add an additional text input for context, regardless of input type
user_context = st.text_area(
    "Provide additional context or focus points (e.g., 'Generate test cases only for the login form' or 'This PDF is about user authentication'):",
    height=100,
    placeholder="e.g., Generate test cases for the 'Forgot Password' functionality."
)

if input_type == "Image (UI Screenshot)":
    uploaded_file = st.file_uploader(
        "Upload a UI screenshot", 
        type=["png", "jpg", "jpeg"],
        help="Upload an image of the user interface you want to test."
    )
elif input_type == "PDF (Requirements Doc)":
    uploaded_file = st.file_uploader(
        "Upload a requirements PDF", 
        type="pdf",
        help="Upload a PDF file containing project requirements or specifications."
    )
else: # Text
    user_text = st.text_area(
        "Paste your user story or requirements text here:", 
        height=250,
        placeholder="As a user, I want to be able to log in with my email and password so that I can access my dashboard."
    )

# Step 3: Generate Test Cases
st.subheader("3. Generate Test Cases")
generate_button = st.button("✨ Generate Now", type="primary", use_container_width=True)

# Divider
st.markdown("---")

# Step 4: Review Generated Test Cases
st.subheader("4. Review Generated Test Cases")

# Use session state to store the results
if 'response_text' not in st.session_state:
    st.session_state.response_text = ""
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""

if generate_button:
    # --- Pre-generation Checks ---
    if not api_key:
        st.warning("Please enter your Google AI API Key in the sidebar to proceed.", icon="🔑")
        st.stop()

    content_to_process = None
    
    # --- System Prompt (Updated) ---
    sys_prompt = f"""
    You are an expert Senior QA Engineer. Your goal is to generate comprehensive test cases based on the provided input (UI screenshot, PDF, or text) and the user's additional context.

    First, provide a one-line summary of the test cases you have generated. The summary should be concise and at the top of your response. For example: "Generated test cases for the user login functionality." or "Test cases for the e-commerce product page based on the provided PDF."

    The test cases you generate MUST be written in simple, clear, and easy-to-understand words.
    
    Your output must cover all possible categories, including:
    - UI/UX Test Cases
    - Positive Scenarios 
    - Negative Scenarios (Error Handling)
    - Functional Test Cases
    - Edge Cases
    - Accessibility Test Cases

    CRITICAL REQUIREMENT:
    You MUST format the test cases as a single, clean Markdown table directly following the summary. The table MUST have the following columns:
    - **Test Case ID**: A unique identifier (e.g., TC-UI-001).
    - **Category**: The type of test (e.g., UI/UX, Functional).
    - **Test Scenario / Description**: A clear, simple description of the test step.
    - **Expected Result**: The expected outcome of the test.

    Do not include any other introductory text, explanations, or summaries outside of the one-line summary and the Markdown table. Your response should begin directly with the summary, followed by the table.
    """

    # Create the prompt based on the input type and user context
    if user_context.strip():
        base_prompt = f"Generate test cases for the following input. Focus on the user's request: '{user_context.strip()}'."
    else:
        base_prompt = "Generate test cases based on the following input."

    if input_type == "Image (UI Screenshot)":
        if uploaded_file is None:
            st.warning("Please upload a UI screenshot before generating.", icon="⚠️")
            st.stop()
        image = Image.open(uploaded_file)
        content_to_process = [base_prompt, image]

    elif input_type == "PDF (Requirements Doc)":
        if uploaded_file is None:
            st.warning("Please upload a requirements PDF before generating.", icon="⚠️")
            st.stop()
        pdf_text = process_pdf(uploaded_file)
        if not pdf_text:
            st.error("Failed to extract text from the PDF. Please try a different file.", icon="🚨")
            st.stop()
        content_to_process = [f"{base_prompt}\n\nPDF Content:\n{pdf_text}"]

    elif input_type == "Text (User Story)":
        if not user_text.strip():
            st.warning("Please paste a user story or requirements text before generating.", icon="⚠️")
            st.stop()
        content_to_process = [f"{base_prompt}\n\nUser Story/Text:\n{user_text}"]
    
    else:
        # This case is a fallback and likely won't be reached
        st.warning("Please provide an input (Image, PDF, or Text) before generating.", icon="⚠️")
        st.stop()
    
    # --- AI Model Interaction ---
    if content_to_process:
        with st.spinner("🤖 AI is analyzing the input and crafting test cases..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_name, 
                    system_instruction=sys_prompt
                )
                response = model.generate_content(content_to_process)
                full_response_text = response.text
                
                # Parse the summary and the table from the response
                summary, table_text = parse_response(full_response_text)
                
                st.session_state.summary_text = summary
                st.session_state.response_text = table_text

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
                st.session_state.response_text = ""
                st.session_state.summary_text = ""

# Display the results if they exist in session state
if st.session_state.response_text:
    # Display the summary first
    if st.session_state.summary_text:
        st.info(f"Summary: {st.session_state.summary_text}")
        
    # Display the markdown table
    st.markdown(st.session_state.response_text)

    # --- Dataframe Conversion and Download ---
    try:
        table_lines = st.session_state.response_text.strip().split('\n')
        
        header_index = -1
        for i, line in enumerate(table_lines):
            if '|' in line and '---' in table_lines[i+1]:
                header_index = i
                break
        
        if header_index != -1:
            header = [h.strip() for h in table_lines[header_index].split('|') if h.strip()]
            data = []
            for line in table_lines[header_index + 2:]:
                if '|' in line:
                    row = [r.strip() for r in line.split('|')]
                    # Clean up empty strings from split
                    if row and row[0] == '': row.pop(0)
                    if row and row[-1] == '': row.pop(-1)
                    if len(row) == len(header):
                        data.append(row)

            if data:
                df = pd.DataFrame(data, columns=header)
                st.success("Successfully parsed test cases into a table.", icon="✅")
                
                csv_data = convert_df_to_csv(df)
                st.download_button(
                    label="📥 Download Test Cases (CSV)",
                    data=csv_data,
                    file_name="generated_test_cases.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("Could not parse the AI's response into a table. You can copy the text above.", icon="📋")
        else:
            st.warning("Could not find a valid Markdown table in the AI's response.", icon="📋")

    except Exception as e:
        st.error(f"Failed to process the response into a downloadable file. Error: {e}", icon="🚨")
else:

    st.info("The generated test cases will appear here after you click the generate button.")
