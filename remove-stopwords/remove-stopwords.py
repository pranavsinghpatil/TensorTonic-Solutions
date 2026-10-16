def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    r = []
    for i in tokens:
        if i not in stopwords:
            r.append(i)

    return r