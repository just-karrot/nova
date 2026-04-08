# NovaTech Support Agent

A customer support agent application utilizing a Python FastAPI backend and a standalone HTML frontend. The backend loads a local knowledge base, indexes it into a vector database, and generates responses using the Groq API with the `llama-3.3-70b-versatile` model.

## Architecture

*   **Backend:** FastAPI server (`backend/main.py`) handling HTTP requests, query context retrieval via ChromaDB, and Groq API integration using the `llama-3.3-70b-versatile` model.
*   **Frontend:** Independent HTML, CSS, and JavaScript interface (`frontend/index.html`).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Environment Configuration:**
    Create a `.env` file in the `backend/` directory containing your Groq API key:
    ```
    GROQ_API_KEY="your_api_key_here"
    ```
    You can obtain a Groq API key from https://console.groq.com.

3.  **Install Dependencies:**
    Navigate to the backend directory and install the required Python packages:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the Backend Server:**
    From the `backend/` directory, start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be served at `http://127.0.0.1:8000`.

2.  **Start the Frontend:**
    Locate the `frontend/index.html` file and open it in a web browser. No local web server or build process is required.

## API Usage

*   `GET /`: Basic connection test.
*   `GET /health`: Returns the status of the server and confirms the indexed knowledge base sections.
*   `POST /ask`: The primary chat endpoint. Requires a JSON payload matching the `TicketRequest` schema, containing the customer message, name, and history array. Returns a generated response from the model wrapped safely in the correct `response`, `customer`, and `status` fields.
