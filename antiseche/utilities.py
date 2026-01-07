# Auto-generated utilities module
# Contains markdown processing and display functions

import re


def strip_markdown(text):
    """Strip basic markdown syntax from text."""
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove links
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Remove bold
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
    # Remove italic
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove lists
    text = re.sub(r"^[\s]*[-\*\+]\s*", "", text, flags=re.MULTILINE)
    return text


def display_text(text, max_lines=20):
    """Display text in chunks for limited screen."""
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
        print(chunk)
        if i + max_lines < len(lines):
            input("Press Enter to continue...")
