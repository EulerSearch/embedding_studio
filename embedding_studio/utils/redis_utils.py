def ft_escape_punctuation(text: str) -> str:
    """
    Escape RediSearch FT (full text) search punctuation characters
    Based on the RedisSearch punctuation handling in toksep.h
    """
    if not text:
        return text

    # Define all the special characters that need escaping
    escape_chars = {
        " ": "\\ ",
        "\t": "\\\t",
        ",": "\\,",
        ".": "\\.",
        "/": "\\/",
        "(": "\\(",
        ")": "\\)",
        "{": "\\{",
        "}": "\\}",
        "[": "\\[",
        "]": "\\]",
        ":": "\\:",
        ";": "\\;",
        "\\": "\\\\",
        "~": "\\~",
        "!": "\\!",
        "@": "\\@",
        "#": "\\#",
        "$": "\\$",
        "%": "\\%",
        "^": "\\^",
        "&": "\\&",
        "*": "\\*",
        "-": "\\-",
        "=": "\\=",
        "+": "\\+",
        "|": "\\|",
        "'": "\\'",
        "`": "\\`",
        '"': '\\"',
        "<": "\\<",
        ">": "\\>",
        "?": "\\?",
        "_": "\\_",
        "\0": "\\x00",
    }

    # Create a result by replacing each character as needed
    result = ""
    for char in text:
        result += escape_chars.get(char, char)

    return result


def ft_unescape_punctuation(escaped_text: str) -> str:
    """
    Inverse of ft_escape_punctuation. Restores the original characters that
    were escaped for Redis full-text search.
    """

    # The same mapping from the escape function, reversed.
    # Key = what ft_escape_punctuation produces, Value = original character
    # (plus the special case for \x00 => '\0')
    unescape_map = {
        "\\ ": " ",
        "\\\t": "\t",
        "\\,": ",",
        "\\.": ".",
        "\\/": "/",
        "\\(": "(",
        "\\)": ")",
        "\\{": "{",
        "\\}": "}",
        "\\[": "[",
        "\\]": "]",
        "\\:": ":",
        "\\;": ";",
        "\\\\": "\\",
        "\\~": "~",
        "\\!": "!",
        "\\@": "@",
        "\\#": "#",
        "\\$": "$",
        "\\%": "%",
        "\\^": "^",
        "\\&": "&",
        "\\*": "*",
        "\\-": "-",
        "\\=": "=",
        "\\+": "+",
        "\\|": "|",
        "\\'": "'",
        "\\`": "`",
        '\\"': '"',
        "\\<": "<",
        "\\>": ">",
        "\\?": "?",
        "\\_": "_",
        "\\x00": "\0",  # special case for the NULL character
    }

    result = []
    i = 0
    n = len(escaped_text)

    while i < n:
        # If we see a backslash, attempt to interpret it as an escaped sequence.
        if escaped_text[i] == "\\" and i + 1 < n:
            # Special check for "\x00" (4 chars)
            if escaped_text[i : i + 4] == "\\x00":
                # It's the NULL-escape sequence
                result.append("\0")
                i += 4
                continue

            # Otherwise, check two characters: e.g. "\,", "\.", "\(", etc.
            maybe_esc = escaped_text[i : i + 2]
            # If recognized, append the unescaped char to result
            if maybe_esc in unescape_map:
                result.append(unescape_map[maybe_esc])
                i += 2
            else:
                # Not recognized as a valid escape. We'll just treat the
                # '\' as a literal or skip it. Here, let's skip the backslash
                # and directly append the next character.
                # Another approach is to keep the slash (result.append('\\')), etc.
                result.append(escaped_text[i + 1])
                i += 2
        else:
            # Regular character: just append.
            result.append(escaped_text[i])
            i += 1

    return "".join(result)
