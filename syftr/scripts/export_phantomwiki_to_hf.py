"""This script exports the PhantomWiki dataset to a format compatible with Hugging Face datasets."""

import json
from pathlib import Path


def process_dataset(folder: Path):
    """Process a single dataset folder and convert it to Hugging Face format."""
    # Load the articles and questions
    articles_path = folder / "articles.json"
    questions_path = folder / "questions.json"

    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    for article in articles:
        title = article["title"]
        body = article["article"]
        # Create the dataset entry in Hugging Face format
        dataset_entry = {
            "title": title,
            "body": body,
        }

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for question in questions:
        question_text = question["question"]
        answer = question["answer"]
        id = question["id"]
        difficulty = question["difficulty"]
        # Create the dataset entry in Hugging Face format
        dataset_entry = {
            "id": id,
            "question": question_text,
            "answer": answer,
            "difficulty": difficulty,
        }

        # Save the dataset entry in Hugging Face format
        output_file = folder / f"{title.replace(' ', '_')}_dataset.json"
        with open(output_file, "a", encoding="utf-8") as out_f:
            json.dump(dataset_entry, out_f)
            out_f.write("\n")


def main():

    # path to the PhantomWiki dataset
    # assume that the dataset has a number of subfolders each containing a dataset
    # each subfolder:
    # articles.json: this is the grounding dataset
    # questions.json: this has the question answer pairs
    phantomwiki_path = Path(
        "/Users/debadeepta.dey/phantomwiki_generated_datasets/trying_021"
    )

    # folder to save the dataset in huggingface format
    output_path = Path("data/phantomwiki_hf/")

    for subfolder in phantomwiki_path.iterdir():
        if subfolder.is_dir():
            print(f"Processing {subfolder.name}...")
            process_dataset(subfolder)


if __name__ == "__main__":

    main()
