# EXPLANATION.md — `app.py` line by line

This document walks through every part of [app.py](app.py) so someone new to
AI can understand exactly what each piece does and why it's there.

---

## The docstring (top of file)
```python
"""Tiny Agentic AI demo: classifies a user's comment as positive or negative,
remembers it in a vector database, and recalls similar past comments."""
```
A description of the whole file. It's the first thing anyone opens this file
will read, so it states the app's purpose in one sentence.

## Imports
```python
import chromadb
from transformers import pipeline
```
- `chromadb` is the **vector database** library. A vector database stores
  pieces of text as lists of numbers ("embeddings" / "vectors") that capture
  their *meaning*, so it can later find text with similar meaning — not just
  matching keywords.
- `transformers` is HuggingFace's library for using pre-trained AI models.
  `pipeline` is a helper that wraps a model and tokenizer together so you can
  use it with a single function call.

## Loading the sentiment model
```python
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
```
This downloads (on first run) and loads a small HuggingFace model that has
already been trained to read English text and decide whether it sounds
POSITIVE or NEGATIVE. `pipeline("sentiment-analysis", ...)` gives us a ready
function: feed it text, get back a label and a confidence score.

## Setting up the vector database
```python
vector_db_client = chromadb.PersistentClient(path="./vector_db")
comment_collection = vector_db_client.get_or_create_collection(name="agentic_ai_comments")
```
- `PersistentClient(path="./vector_db")` opens (or creates) a database that
  saves its data to the `vector_db/` folder on disk, so the agent's memory
  survives between runs.
- `get_or_create_collection(name=...)` opens (or creates) a named "table" of
  documents inside that database — this is where every comment the agent
  sees will be stored as a vector, alongside metadata like its sentiment.

## `classify_comment`
```python
def classify_comment(comment_text):
    """Ask the HuggingFace model whether the comment is positive or negative."""
    result = sentiment_classifier(comment_text)[0]
    label = "positive" if result["label"] == "POSITIVE" else "negative"
    confidence = round(result["score"] * 100, 1)
    return label, confidence
```
- `sentiment_classifier(comment_text)` runs the HuggingFace model on the
  user's text. It returns a list with one dictionary, e.g.
  `[{"label": "POSITIVE", "score": 0.998}]` — `[0]` grabs that dictionary.
- The model's raw label is `"POSITIVE"`/`"NEGATIVE"` (uppercase); we convert
  it to a friendlier lowercase `"positive"`/`"negative"` for display.
- `result["score"]` is a probability between 0 and 1 (how confident the model
  is). Multiplying by 100 and rounding to 1 decimal turns it into a readable
  percentage like `99.8`.
- The function returns both the label and the confidence so the caller can
  show the user how sure the agent is, not just its verdict.

## `recall_similar_comments`
```python
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
```
- `comment_collection.count()` tells us how many comments are stored so far.
  If it's `0`, there's nothing to compare against, so we return an empty list
  immediately — this avoids asking Chroma for results it can't give.
- `comment_collection.query(query_texts=[comment_text], n_results=...)` is the
  core "vector search": Chroma turns `comment_text` into a vector, compares it
  to every stored comment's vector, and returns the closest matches — i.e.
  comments that *mean* something similar, even if the wording differs.
- `n_results=min(max_results, comment_collection.count())` asks for at most
  `max_results` (3) matches, but never more than the number of comments that
  actually exist (otherwise Chroma would raise an error for a tiny database).
- The query returns parallel lists of matched documents and their metadata,
  nested inside an extra list (because Chroma supports searching for several
  queries at once — we only ever send one, hence `[0]`).
- The `zip(...)` loop pairs each returned comment with its stored sentiment
  (`past_metadata["sentiment"]`) and collects them as `(text, sentiment)`
  tuples to hand back to the caller.

## `remember_comment`
```python
def remember_comment(comment_text, label):
    """Store the comment and its sentiment so future searches can find it."""
    new_id = str(comment_collection.count())
    comment_collection.add(
        documents=[comment_text],
        metadatas=[{"sentiment": label}],
        ids=[new_id],
    )
```
- Every item stored in a Chroma collection needs a unique ID. Using the
  current count as the ID (`"0"`, `"1"`, `"2"`, ...) is a simple way to get a
  fresh, unused ID each time.
- `comment_collection.add(...)` converts `comment_text` into a vector behind
  the scenes and stores it together with `documents` (the original text) and
  `metadatas` (extra structured info — here, the sentiment label). This is
  what gives the agent "memory": the next call to `recall_similar_comments`
  will be able to find this comment.

## `explain_agentic_ai`
```python
def explain_agentic_ai():
    print("=" * 60)
    print("What is Agentic AI?")
    ...
```
Purely educational output, printed once when the app starts. It tells a
newcomer what an "agent" is in plain language and explicitly connects that
definition to what this app is doing (using a model to reason, and a vector
database to remember/recall).

## `run` — the main loop
```python
def run():
    explain_agentic_ai()

    while True:
        comment_text = input("\nEnter a comment (or 'quit' to exit): ").strip()
        if comment_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not comment_text:
            continue
```
- Prints the explanation, then enters an infinite loop that keeps prompting
  the user for comments.
- `input(...)` displays the prompt and waits for the user to type something;
  `.strip()` removes accidental leading/trailing whitespace.
- Typing `quit` or `exit` (in any capitalization) ends the loop and the
  program. Typing nothing (`""`) just re-prompts (`continue`) without doing
  any work.
```python
        label, confidence = classify_comment(comment_text)
        print(f"\nAgent's verdict: this comment is {label.upper()} ({confidence}% confident)")
```
Runs the HuggingFace model on the typed comment and prints the agent's
positive/negative verdict along with how confident it is.
```python
        similar_comments = recall_similar_comments(comment_text)
        if similar_comments:
            print("This reminds the agent of earlier comments:")
            for past_text, past_label in similar_comments:
                print(f"  - \"{past_text}\" was {past_label}")
        else:
            print("The agent has no earlier comments to compare this with yet.")
```
Searches the vector database for similar past comments and prints them with
their previously recorded sentiment — this is the "recall" part of the agent
loop, made visible to the user. If the database is still empty, it says so.
```python
        remember_comment(comment_text, label)
```
Finally, stores the new comment (and its freshly computed sentiment) in the
vector database, so it can be recalled the next time someone enters something
similar — closing the perceive → reason → remember loop.

## Entry point
```python
if __name__ == "__main__":
    run()
```
Standard Python idiom: only start the app when this file is run directly
(e.g. `python app.py`), not when it's imported by something else.
