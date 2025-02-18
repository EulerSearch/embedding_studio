from typing import List


def generate_fuzzy_regex(text: str) -> str:
    if not text:
        return ""
    if len(text) == 1:
        return f"^[{text.upper()}{text.lower()}]"

    patterns = []

    # Generate patterns with substitutions at different positions
    for i in range(len(text)):
        pattern = []
        # Before substitution
        if i > 0:
            for j in range(i):
                pattern.append(f"[{text[j].upper()}{text[j].lower()}]")

        # Substitution part
        pattern.append("[a-zA-Z]{1,2}")

        # After substitution
        for j in range(i + 1, len(text)):
            pattern.append(f"[{text[j].upper()}{text[j].lower()}]")

        patterns.append("".join(pattern))

    # Combine all patterns
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
