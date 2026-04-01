# Intern Support Chatbot (NLP)

AI chatbot project for internship support that answers intern queries in real time using FAQ documents and historical support tickets.

## Objective

Build an NLP-based chatbot to automate first-level responses for intern questions and reduce manual support workload.

## Features

- Real-time question answering for interns
- Uses FAQ and ticket-resolution knowledge base
- Intent and confidence output
- Confidence-based fallback to human support
- Simple command-line chat interface

## Tech Stack

- Python 3.9+
- scikit-learn (`TfidfVectorizer`, cosine similarity)
- NumPy

## Project Structure

```text
.
├── intern_support_chatbot.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
└── data
    ├── faqs.json
    └── tickets.json
```

## Dataset Format

### FAQ file (`data/faqs.json`)

List of objects with:
- `question` (string)
- `answer` (string)
- `intent` (string, optional)

### Ticket file (`data/tickets.json`)

List of objects with:
- `query` or `question` (string)
- `resolution` or `answer` (string)
- `intent` (string, optional)

## Setup

1. Create and activate virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

### Run with built-in fallback/default data

```bash
python intern_support_chatbot.py
```

### Run with provided sample dataset

```bash
python intern_support_chatbot.py --faq data/faqs.json --tickets data/tickets.json
```

### Optional confidence threshold

```bash
python intern_support_chatbot.py --faq data/faqs.json --tickets data/tickets.json --threshold 0.35
```

## Example Output

```text
=== Intern Support Chatbot ===
Knowledge items loaded: 12
Internal intent accuracy: 100.00%
Average confidence: 0.911
Type 'exit' to quit.
```

## Internship Outcome

This project demonstrates:
- Practical NLP implementation for support automation
- Knowledge retrieval from organizational documents
- Human handoff strategy when model confidence is low

## Future Improvements

- Replace TF-IDF retriever with sentence-transformer embeddings
- Add Rasa dialogue manager for multi-turn flows
- Build a web UI (Streamlit/Flask/FastAPI)
- Add logging and analytics dashboard for unresolved queries

## Author

Internship Final Project by **[Your Name]**.
