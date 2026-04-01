"""
Intern Support Chatbot - Internship Final Project Prototype

Objective:
Build an AI chatbot to answer intern queries in real time using FAQ data and
historical support tickets.

This script provides a practical baseline:
- Data sources: FAQ documents and historical ticket logs (CSV/JSON).
- NLP model: TF-IDF retrieval + intent prediction via nearest-neighbor labels.
- Outcome: real-time automated responses with confidence-based fallback.

How to run:
1) (Optional) Create a virtual environment and install dependencies:
   pip install scikit-learn numpy

2) Run:
   python intern_support_chatbot.py

3) Optional custom data files:
   python intern_support_chatbot.py --faq data/faqs.json --tickets data/tickets.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeItem:
    question: str
    answer: str
    intent: str
    source: str


def normalize_text(text: str) -> str:
    """Basic text cleanup for robust matching."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json_items(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def load_csv_items(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({k: (v or "").strip() for k, v in row.items()})
    return items


def load_knowledge_base(
    faq_path: Optional[Path] = None, tickets_path: Optional[Path] = None
) -> List[KnowledgeItem]:
    """
    Load knowledge items from FAQ and ticket files.

    Expected formats:
    - FAQ JSON/CSV with fields: question, answer, intent(optional)
    - Tickets JSON/CSV with fields: query(or question), resolution(or answer), intent(optional)
    """
    kb: List[KnowledgeItem] = []

    # Default starter FAQs if no files are provided.
    default_faqs = [
        {
            "question": "How do I submit my daily internship report?",
            "answer": "Submit your daily report by 6 PM in the internship portal under 'Daily Logs'.",
            "intent": "report_submission",
        },
        {
            "question": "What is the attendance policy for interns?",
            "answer": "Interns must maintain at least 85% attendance and mark check-in/check-out daily.",
            "intent": "attendance_policy",
        },
        {
            "question": "Who do I contact for technical issues?",
            "answer": "For technical issues, contact the support mentor via the #tech-support channel.",
            "intent": "technical_support",
        },
        {
            "question": "How can I request leave during internship?",
            "answer": "Submit leave requests through the HR form at least 24 hours in advance.",
            "intent": "leave_request",
        },
        {
            "question": "When is the final project submission deadline?",
            "answer": "The final project deadline is in Week 6, Sunday 11:59 PM.",
            "intent": "deadline_query",
        },
    ]

    if faq_path and faq_path.exists():
        faq_items = (
            load_json_items(faq_path)
            if faq_path.suffix.lower() == ".json"
            else load_csv_items(faq_path)
        )
    else:
        faq_items = default_faqs

    for item in faq_items:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question or not answer:
            continue
        kb.append(
            KnowledgeItem(
                question=question,
                answer=answer,
                intent=item.get("intent", "general_faq").strip() or "general_faq",
                source="faq",
            )
        )

    if tickets_path and tickets_path.exists():
        ticket_items = (
            load_json_items(tickets_path)
            if tickets_path.suffix.lower() == ".json"
            else load_csv_items(tickets_path)
        )
        for item in ticket_items:
            query = item.get("query", "").strip() or item.get("question", "").strip()
            resolution = item.get("resolution", "").strip() or item.get("answer", "").strip()
            if not query or not resolution:
                continue
            kb.append(
                KnowledgeItem(
                    question=query,
                    answer=resolution,
                    intent=item.get("intent", "ticket_support").strip() or "ticket_support",
                    source="ticket",
                )
            )

    if not kb:
        raise ValueError("Knowledge base is empty. Provide valid FAQ/ticket data.")

    return kb


class InternSupportChatbot:
    """A lightweight retrieval-based chatbot for intern support."""

    def __init__(self, confidence_threshold: float = 0.30) -> None:
        self.confidence_threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.kb: List[KnowledgeItem] = []
        self.question_matrix = None

    def train(self, kb: List[KnowledgeItem]) -> None:
        self.kb = kb
        questions = [normalize_text(item.question) for item in kb]
        self.question_matrix = self.vectorizer.fit_transform(questions)

    def predict(self, user_query: str) -> Dict[str, str]:
        if self.question_matrix is None or not self.kb:
            raise RuntimeError("Model not trained. Call train() first.")

        clean_query = normalize_text(user_query)
        query_vec = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vec, self.question_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_item = self.kb[best_idx]

        if best_score < self.confidence_threshold:
            return {
                "response": (
                    "I am not fully confident about that query. "
                    "Please contact your mentor/support team for exact guidance."
                ),
                "intent": "fallback",
                "confidence": f"{best_score:.3f}",
                "source": "fallback",
            }

        return {
            "response": best_item.answer,
            "intent": best_item.intent,
            "confidence": f"{best_score:.3f}",
            "source": best_item.source,
        }


def evaluate(chatbot: InternSupportChatbot, kb: List[KnowledgeItem]) -> Tuple[float, float]:
    """
    Quick internal evaluation:
    - Intent accuracy (predicted intent on known questions)
    - Mean confidence score
    """
    correct = 0
    confidences: List[float] = []
    for item in kb:
        pred = chatbot.predict(item.question)
        confidences.append(float(pred["confidence"]))
        if pred["intent"] == item.intent:
            correct += 1
    intent_acc = correct / len(kb)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return intent_acc, avg_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intern Support Chatbot (FAQ + ticket based NLP chatbot)"
    )
    parser.add_argument("--faq", type=str, default="", help="Path to FAQ JSON/CSV")
    parser.add_argument(
        "--tickets", type=str, default="", help="Path to ticket history JSON/CSV"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for fallback response",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    faq_path = Path(args.faq) if args.faq else None
    tickets_path = Path(args.tickets) if args.tickets else None

    kb = load_knowledge_base(faq_path=faq_path, tickets_path=tickets_path)
    bot = InternSupportChatbot(confidence_threshold=args.threshold)
    bot.train(kb)

    intent_acc, avg_conf = evaluate(bot, kb)
    print("=== Intern Support Chatbot ===")
    print(f"Knowledge items loaded: {len(kb)}")
    print(f"Internal intent accuracy: {intent_acc:.2%}")
    print(f"Average confidence: {avg_conf:.3f}")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Intern > ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Chatbot > Goodbye! Best of luck with your internship project.")
            break
        if not user_query:
            continue

        pred = bot.predict(user_query)
        print(f"Chatbot > {pred['response']}")
        print(
            f"[intent={pred['intent']} | confidence={pred['confidence']} | source={pred['source']}]"
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
