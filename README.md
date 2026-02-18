# 🔍 Truth Seeker — AI-Powered Claim Verification Assistant

A Streamlit-based fact-checking application that uses **RAG (Retrieval-Augmented Generation)** to verify claims against a curated database of Snopes fact-check articles. The system retrieves relevant context using FAISS vector search and generates structured verdicts with probability scores via GPT-4o.

## Demo

![App Screenshot](assets/demo_screenshot.png)

## How It Works

1. **Data Ingestion** — Fact-check articles are loaded from a curated JSON dataset sourced from Snopes (technology category).
2. **Chunking & Embedding** — Articles are split using `RecursiveCharacterTextSplitter` and embedded with OpenAI embeddings.
3. **Vector Store** — Chunks are indexed in a FAISS vector store for fast similarity search.
4. **Retrieval** — When a user submits a claim, the top-3 most relevant chunks are retrieved.
5. **LLM Verdict** — GPT-4o evaluates the claim against the retrieved context and returns a structured output with probabilities for `True`, `False`, and `Unproven`, along with a detailed rationale.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | GPT-4o (via LangChain) |
| Embeddings | OpenAI Embeddings |
| Vector Store | FAISS |
| Output Parsing | Pydantic + LangChain |

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/truth-seeker-fact-checker.git
cd truth-seeker-fact-checker

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run the App

```bash
streamlit run app.py
```

## Project Structure

```
.
├── app.py                          # Streamlit application (main entry point)
├── vector_stores.py                # Data loading, chunking, FAISS indexing, and retrieval
├── read.py                         # Utility script for data exploration / preprocessing
├── technology_fact_checks.json     # Curated fact-check dataset (Snopes — technology)
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── BUDT758O_Final Presentation.pdf # Project presentation
```

## Usage

1. Launch the app with `streamlit run app.py`
2. Enter any claim in the text area (e.g., *"5G towers spread COVID-19"*)
3. Click **Check Claim**
4. View the verdict (True / False / Unproven), confidence scores, probability distribution chart, rationale, and source links

## Course

**BUDT758O — Designing Generative AI Systems**  
M.S. Information Systems, University of Maryland

## License

This project was developed for academic purposes. The fact-check data is sourced from [Snopes](https://www.snopes.com/).
