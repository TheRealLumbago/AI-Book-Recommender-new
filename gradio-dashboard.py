import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import re

# Load environment variables
load_dotenv("C:/Users/Ali/PycharmProjects/book-recommender/.venv/.env")

# Load books data
books = pd.read_csv("C:/Users/Ali/PycharmProjects/book-recommender/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load text data and prepare embeddings
raw_documents = TextLoader("C:/Users/Ali/PycharmProjects/book-recommender/tagged_descriptions.txt",
                           encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


# Retrieve semantic recommendations
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract book ISBNs from the embeddings search results
    books_list = [int(re.sub(r'\D', '', rec.page_content.strip('"').split()[0])) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Filter by category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by emotional tone if specified
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


# Generate book recommendations
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Use the full description without truncation
        description = row["description"]

        # Format authors
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Caption format
        title = row["title"]
        caption = f"<strong>{title}</strong> by {authors_str}<br><br>{description}"

        # Append thumbnail URL and caption to results
        results.append((row["large_thumbnail"], caption))

    return results



# Define categories and tones for dropdown
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    # Header with "Created by Ali Chaudhry"
    with gr.Row():
        gr.Markdown("# Book Recommender")
        gr.HTML(
            "<div style='text-align: right; margin-left: auto; font-size: 0.9rem;'>"
            "Created by Ali Chaudhry</div>"
        )

    # Input section
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("Find Recommendations")

    # Recommendations section
    gr.Markdown("## Recommendations")
    with gr.Column():
        output_gallery = gr.HTML(label="Recommended Books")

    submit_button.click(
        fn=lambda query, category,
                  tone: "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>"
                        + "".join(
            f"<div style='display: flex; align-items: flex-start; margin-bottom: 1rem;'>"
            f"<img src='{thumbnail}' alt='Thumbnail' style='width: 100px; height: auto; margin-right: 1rem;'>"
            f"<div>{caption}</div></div>"
            for thumbnail, caption in recommend_books(query, category, tone)
        ) + "</div>",
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[output_gallery]
    )

# Launch the app
if __name__ == "__main__":
    dashboard.launch(share=True)
