def is_empty_string(argument: str) -> bool:
    """
    A function to check if argument is an empty string

    Args:
        argument (str): string to be checked

    Returns:
        bool: True if argument is an empty string
    """
    if not isinstance(argument, str) or argument is None or not argument:
        return True

    else:
        return len(argument.strip()) <= 0