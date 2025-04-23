from typing import List


def generate_fuzzy_regex(text: str) -> str:
    """
    Generate a fuzzy regex pattern that matches variations of the input text.

    This function creates a regular expression pattern that can match the input text
    with one character potentially replaced by 1-2 other characters. This is useful
    for fuzzy text matching where exact matches are not required.

    The generated pattern:
    1. Is case-insensitive (matches both upper and lowercase variants)
    2. Allows for 1-2 character substitutions at any position
    3. Is anchored to the start of the string with '^'

    :param text: The input text to generate a fuzzy regex pattern for
    :return: A regex pattern string that matches variations of the input text
    """
    # Handle empty input case
    if not text:
        return ""

    # Handle single character input case
    if len(text) == 1:
        # For a single character, just match that character case-insensitively
        return f"^[{text.upper()}{text.lower()}]"

    patterns = []

    # Generate patterns with substitutions at different positions
    for i in range(len(text)):
        pattern = []
        # Before substitution: match original characters case-insensitively
        if i > 0:
            for j in range(i):
                # For each character before the substitution point,
                # create a character class matching both upper and lowercase
                pattern.append(f"[{text[j].upper()}{text[j].lower()}]")

        # Substitution part: allow any 1-2 alphabetic characters
        pattern.append("[a-zA-Z]{1,2}")

        # After substitution: match original characters case-insensitively
        for j in range(i + 1, len(text)):
            # For each character after the substitution point,
            # create a character class matching both upper and lowercase
            pattern.append(f"[{text[j].upper()}{text[j].lower()}]")

        # Join the parts into a complete pattern for this substitution position
        patterns.append("".join(pattern))

    # Combine all patterns with alternation (|) and anchor to start of string
    combined_pattern = f"^({'|'.join(patterns)})"
    return combined_pattern


def combine_chunks(chunks: List[str]) -> str:
    """
    Combine tokens into a single string, merging punctuation appropriately.

    Rules in this example:
      - If a chunk is in punctuation_attach_left (like . ! ? , etc.),
        it is appended to the previous token with no space.
      - If a chunk is in punctuation_attach_both (like -),
        it merges with both the previous token and the *next* token.
      - All other tokens are separated by spaces.

    Example:
      chunks = ['a', 'b', '-', 'c', 'd', '.']
      returns "a b-c d."
    """
    chunks = [chunk for chunk in chunks]
    punctuation_attach_left = {".", ",", "!", "?", ":", ";", ")", "]", "}"}
    punctuation_attach_both = {"-", "/"}  # hyphens, slashes, etc.

    result: List[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]

        # 1) Punctuation that attaches to the token on its LEFT
        #    e.g., . , ! ? => "word."
        if chunk in punctuation_attach_left:
            if result:
                result[-1] += chunk
            else:
                # No preceding token—just add it (edge case)
                result.append(chunk)

        # 2) Punctuation that merges with BOTH sides
        #    e.g., '-' => "word-other"
        elif chunk in punctuation_attach_both:
            if result and (i + 1) < len(chunks):
                # Merge chunk with the last token in result + next chunk
                next_chunk = chunks[i + 1]
                result[-1] += chunk + next_chunk
                # Skip the next token since we merged it
                i += 1
            else:
                # Edge case: no next token or no previous token
                if not result:
                    result.append(chunk)
                else:
                    result[-1] += chunk

        # 3) Everything else (normal tokens)
        else:
            result.append(chunk)

        i += 1

    # Join the final tokens with a space
    return " ".join(result)
