"""Splitting and cleaning the captions in the `description` column.

Each description holds 10 captions separated by newlines. Milestone 2
(captioning) will reuse these functions.

Example:
    clean_captions("This bird is RED .\\n\\nIt has a short,  pointy beak!\\n")
    -> ["this bird is red", "it has a short, pointy beak"]
"""

import re


def split_description(description):
    """Split a description into its captions, skipping empty lines."""
    return [line.strip() for line in description.split("\n") if line.strip()]


def clean_caption(caption):
    """Lowercase a caption and remove the noise we found in the EDA."""
    text = caption.lower()

    # About 100 captions have every space replaced by the broken character "�".
    text = text.replace("�", " ")
    # Curly apostrophes and long dashes -> plain ones.
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("&", " and ").replace("/", " ")

    # Keep only letters, digits, spaces and , . ' -
    text = re.sub(r"[^a-z0-9 ,.'\-]", " ", text)
    # Repeated punctuation: ".." -> "."
    text = re.sub(r"([,.'\-])\1+", r"\1", text)
    # No space before a comma or period: "red ," -> "red,"
    text = re.sub(r"\s+([,.])", r"\1", text)
    # One space after a comma or period: "red,with" -> "red, with"
    text = re.sub(r"([,.])(?=[a-z])", r"\1 ", text)
    # Many spaces -> one space
    text = re.sub(r"\s+", " ", text)

    return text.strip(" ,.'-")


def clean_captions(description):
    """Split a description and clean every caption. Captions that end up empty are dropped."""
    captions = [clean_caption(c) for c in split_description(description)]
    return [c for c in captions if c]


def tokenize(caption):
    """Split a cleaned caption into words (punctuation removed, "bird's" kept as one word)."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", caption)
