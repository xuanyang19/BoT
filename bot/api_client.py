# -*- coding: utf-8 -*-
"""
Generic API client for OpenAI.
Reads API key from OPENAI_API_KEY environment variable.
"""

import os
from openai import OpenAI


def get_openai_client():
    """
    Get OpenAI client using API key from environment.

    Environment variable:
        OPENAI_API_KEY: Your OpenAI API key

    Returns:
        OpenAI client instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    return OpenAI(api_key=api_key)
