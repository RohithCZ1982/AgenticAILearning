"""Tiny Agentic AI demo: classifies a user's comment as positive or negative,
remembers it in a vector database, and recalls similar past comments."""

import chromadb
from transformers import pipeline

# A HuggingFace pipeline that already knows how to label text as POSITIVE/NEGATIVE.
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# A small on-disk vector database. Chroma turns each comment into a vector
# (an embedding) so it can later find comments with similar meaning.
vector_db_client = chromadb.PersistentClient(path="./vector_db")
comment_collection = vector_db_client.get_or_create_collection(name="agentic_ai_comments")


def classify_comment(comment_text):
    """Ask the HuggingFace model whether the comment is positive or negative."""
    result = sentiment_classifier(comment_text)[0]
    label = "positive" if result["label"] == "POSITIVE" else "negative"
    confidence = round(result["score"] * 100, 1)
    return label, confidence


def recall_similar_comments(comment_text, max_results=3):
    """Search the vector database for comments that are similar in meaning."""
    if comment_collection.count() == 0:
        return []

    search_results = comment_collection.query(
        query_texts=[comment_text],
        n_results=min(max_results, comment_collection.count()),
    )

    remembered = []
    for past_text, past_metadata in zip(search_results["documents"][0], search_results["metadatas"][0]):
        remembered.append((past_text, past_metadata["sentiment"]))
    return remembered


def remember_comment(comment_text, label):
    """Store the comment and its sentiment so future searches can find it."""
    new_id = str(comment_collection.count())
    comment_collection.add(
        documents=[comment_text],
        metadatas=[{"sentiment": label}],
        ids=[new_id],
    )


def explain_agentic_ai():
    print("=" * 60)
    print("What is Agentic AI?")
    print("-" * 60)
    print(
        "An 'agent' is an AI program that does more than answer once.\n"
        "It can: (1) perceive input, (2) use tools or models to reason\n"
        "about it, (3) remember what happened, and (4) use that memory\n"
        "to act better next time. This tiny app is an agent because it\n"
        "uses a HuggingFace model to judge sentiment AND a vector\n"
        "database to remember and recall earlier comments."
    )
    print("=" * 60)


def run():
    explain_agentic_ai()

    while True:
        comment_text = input("\nEnter a comment (or 'quit' to exit): ").strip()
        if comment_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not comment_text:
            continue

        label, confidence = classify_comment(comment_text)
        print(f"\nAgent's verdict: this comment is {label.upper()} ({confidence}% confident)")

        similar_comments = recall_similar_comments(comment_text)
        if similar_comments:
            print("This reminds the agent of earlier comments:")
            for past_text, past_label in similar_comments:
                print(f"  - \"{past_text}\" was {past_label}")
        else:
            print("The agent has no earlier comments to compare this with yet.")

        remember_comment(comment_text, label)


if __name__ == "__main__":
    run()
