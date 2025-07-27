RAG-based MCQ & Short Answer Generator
This project is an intelligent study tool that leverages a Retrieval-Augmented Generation (RAG) pipeline to automatically generate multiple-choice questions (MCQs) and short-answer questions from any uploaded PDF document.

Built with Streamlit, it provides a simple user interface where a user can upload a GenAI-related PDF, and the system will produce a set of questions to test their knowledge on the document's content.

🎯 Project Goal
The primary goal is to create an automated question-generation system that helps users study and review technical documents more efficiently. The application should be able to:

Accept a PDF document as input.

Understand the context and key concepts within the document.

Generate 5 relevant MCQs with options and correct answers.

Generate 5 relevant short-answer questions.

🛠️ Tech Stack & Architecture
Application Framework: Streamlit

LLM Orchestration: LangChain

Vector Store: FAISS (Facebook AI Similarity Search)

LLM Provider: Azure OpenAI

Core Language: Python

⚙️ RAG Pipeline Workflow
The project is built around a classic RAG pipeline to ensure the generated questions are grounded in the content of the provided document.

Document Upload: The user uploads a PDF file through the Streamlit web interface.

Content Extraction & Chunking: The system extracts all text from the PDF and splits it into smaller, semantically meaningful chunks. This is crucial for effective retrieval.

Vectorization & Storage: Each text chunk is converted into a numerical vector (embedding) using an embedding model. These vectors are then stored and indexed in a FAISS vector database for fast similarity search.

Retrieval: When a query is made to generate questions, the RetrievalQA chain from LangChain first retrieves the most relevant text chunks from the FAISS vector store.

Generation: The retrieved chunks are passed along with a specific prompt to the Azure OpenAI model. The LLM then uses this context to generate high-quality MCQs and short-answer questions.

Display: The final generated questions and answers are displayed to the user in the Streamlit app.

🏁 How to Run
Prerequisites
Python 3.8+

An Azure OpenAI API Key and endpoint

Installation & Execution
Clone the repository:

git clone https://github.com/upadhyay-jash/mcq-generator.git
cd mcq-generator

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file in the root directory.

Add your Azure OpenAI credentials to the .env file:

AZURE_OPENAI_API_KEY="YOUR_API_KEY"
AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"

Run the Streamlit app:

streamlit run app.py
