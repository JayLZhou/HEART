from hashlib import md5
import re
from colorama import Fore, Style, init
from pyfiglet import Figlet

def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def split_string_by_multi_markers(
        text: str, delimiters: list[str]
) -> list[str]:
    """
    Split a string by multiple delimiters.

    Args:
        text (str): The string to split.
        delimiters (list[str]): A list of delimiter strings.

    Returns:
        list[str]: A list of strings, split by the delimiters.
    """
    if not delimiters:
        return [text]
    split_pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)
    segments = re.split(split_pattern, text)
    return [segment.strip() for segment in segments if segment.strip()]

def welcome_message():
    f = Figlet(font='big')  #
    # Generate the large ASCII art text
    logo = f.renderText('HEART')
    print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
    # Print the logo with color
    print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
    text = [
        "Welcome to HEART: A query-level RAG tuning system.",
        "",
        "Heart is a query-level RAG tuning system that allows you to tune your RAG models for specific queries.",
        "",
        "We hope this will be helpful to you!"
    ]

    # Function to print a boxed message
    def print_box(text_lines, border_color=Fore.BLUE, text_color=Fore.CYAN):
        max_length = max(len(line) for line in text_lines)
        border = f"{border_color}╔{'═' * (max_length + 2)}╗{Style.RESET_ALL}"
        print(border)
        for line in text_lines:
            print(
                f"{border_color}║{Style.RESET_ALL} {text_color}{line.ljust(max_length)} {border_color}║{Style.RESET_ALL}")
        border = f"{border_color}╚{'═' * (max_length + 2)}╝{Style.RESET_ALL}"
        print(border)

    # Print the boxed welcome message
    print_box(text)

    # Add a decorative line for separation
    print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
