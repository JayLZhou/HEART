from hashlib import md5
import re
from colorama import Fore, Style, init
from pyfiglet import Figlet
import shutil
import io
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple
from Common.Logger import logger
from tenacity import RetryCallState

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


def clean_storage(path):
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
                print(f"File {path} has been deleted.")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Directory {path} and its contents have been deleted.")
            else:
                print(f"The path {path} exists but is not a file or directory.")
        else:
            print(f"The path {path} does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting {path}: {e}")

def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"

def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)        

def any_to_str_set(val) -> set:
    """Convert any type to string set."""
    res = set()

    # Check if the value is iterable, but not a string (since strings are technically iterable)
    if isinstance(val, (dict, list, set, tuple)):
        # Special handling for dictionaries to iterate over values
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res

def log_and_reraise(retry_state: RetryCallState):
    logger.error(f"Retry attempts exhausted. Last exception: {retry_state.outcome.exception()}")
    logger.warning(
        """
Recommend going to https://deepwisdom.feishu.cn/wiki/MsGnwQBjiif9c3koSJNcYaoSnu4#part-XdatdVlhEojeAfxaaEZcMV3ZniQ
See FAQ 5.8
"""
    )
    raise retry_state.outcome.exception()

def prase_json_from_response(response: str) -> dict:
    """
    Extract JSON data from a string response.

    This function attempts to extract the first complete JSON object from the response.
    If that fails, it tries to extract key-value pairs from a potentially malformed JSON string.

    Args:
        response: The string response containing JSON data.
    Returns:
        A dictionary containing the extracted JSON data.
    """
    stack = []
    first_json_start = None

    # Attempt to extract the first complete JSON object using a stack to track braces
    for i, char in enumerate(response):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = response[first_json_start:i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        break
                    finally:
                        first_json_start = None

    # If extraction of complete JSON failed, try extracting key-value pairs from a non-standard JSON string
    extracted_values = {}
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, response, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = prase_json_from_response(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value_from_string(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    else:
        logger.info("JSON data successfully extracted.")

    return extracted_values