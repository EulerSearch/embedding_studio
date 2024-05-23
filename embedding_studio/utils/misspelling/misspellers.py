import random
import string

import nltk
from nltk.tokenize.api import TokenizerI

from embedding_studio.utils.misspelling.keyboard_layout import KEYBOARD_LAYOUT


def adjacent_key_error(word: str) -> str:
    """
    Introduce adjacent key errors based on a keyboard layout.

    :param word: The original word.
    :return: The word with adjacent key errors introduced.
    """
    new_word = ""
    for char in word:
        if (
            char in KEYBOARD_LAYOUT and random.random() < 0.1
        ):  # Adjust the probability as needed
            new_word += random.choice(KEYBOARD_LAYOUT[char])
        else:
            new_word += char
    return new_word


def delete_random_character(word: str) -> str:
    """
    Delete a random character from the word.

    :param word: The original word.
    :return: The word with a random character deleted.
    """
    if len(word) > 1:
        index_to_remove = random.randint(0, len(word) - 1)
        return word[:index_to_remove] + word[index_to_remove + 1 :]
    return word


def swap_characters(string: str, i: int, j: int) -> str:
    """
    Swap two characters in a string.

    :param string: The original string.
    :param i: The index of the first character to swap.
    :param j: The index of the second character to swap.
    :return: The string with the characters swapped.
    """
    char_list = list(string)
    char_list[i], char_list[j] = char_list[j], char_list[i]
    return "".join(char_list)


def swap_random_adjacent_characters(string: str) -> str:
    """
    Swap two adjacent characters in a string.

    :param string: The original string.
    :return: The string with two adjacent characters swapped.
    """
    if len(string) < 2:
        return string
    i = random.randint(0, len(string) - 2)
    return swap_characters(string, i, i + 1)


def insert_random_character(word: str) -> str:
    """
    Insert a random character into the word.

    :param word: The original word.
    :return: The word with a random character inserted.
    """
    index_to_insert = random.randint(0, len(word))
    random_char = random.choice(string.ascii_lowercase)
    return word[:index_to_insert] + random_char + word[index_to_insert:]


def random_split(word: str) -> str:
    """
    Randomly split a word into two parts.

    :param word: The original word.
    :return: The word split into two parts with a space in between.
    """
    if len(word) > 1:
        split_index = random.randint(1, len(word) - 1)
        return word[:split_index] + " " + word[split_index:]
    return word


def introduce_misspellings_with_keyboard_map(
    text: str, error_rate: float = 0.1, tokenizer: TokenizerI = None
) -> str:
    """
    Introduce misspellings in a text using a keyboard layout.

    :param text: The original text.
    :param error_rate: The rate of errors to introduce.
    :param tokenizer: The tokenizer to use for tokenizing the text.
    :return: The text with misspellings introduced.
    """
    tokenizer = (
        nltk.TreebankWordTokenizer() if tokenizer is None else tokenizer
    )
    words_and_spans = list(tokenizer.span_tokenize(text))
    tokens = [text[start:end] for start, end in words_and_spans]

    misspelled_tokens = []
    prev_end = 0
    ignore_seprator = False
    for token, (start, _) in zip(tokens, words_and_spans):
        # Extracting the separator using the span information
        separator = text[prev_end:start]
        if not ignore_seprator:
            misspelled_tokens.append(separator)

        ignore_seprator = False

        error_choice = random.random()
        if error_choice < error_rate:
            error_type = random.choice(
                [
                    adjacent_key_error,
                    delete_random_character,
                    swap_random_adjacent_characters,
                    insert_random_character,
                    random_split,
                    None,
                ]
            )
            if error_type is not None:
                misspelled_token = error_type(token)
                misspelled_tokens.append(misspelled_token)
            else:
                misspelled_tokens.append(token)
                ignore_seprator = True
        else:
            misspelled_tokens.append(token)

        prev_end = start + len(token)

    # Append any remaining text after the last token
    misspelled_tokens.append(text[prev_end:])

    misspelled_text = "".join(misspelled_tokens)
    return misspelled_text
