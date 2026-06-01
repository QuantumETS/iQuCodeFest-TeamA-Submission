"""This module is the entry point for your python application."""


def sum(a: int, b: int) -> int:
    """
    Returns the sum of two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of the two integers.

    """
    return a + b


def main():
    """The main function of the application."""
    sum(2, 3)


if __name__ == "__main__":
    main()
