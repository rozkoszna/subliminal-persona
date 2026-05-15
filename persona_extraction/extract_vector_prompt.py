#!/usr/bin/env python3
"""
Prompt-based vector extraction entrypoint.

This is an alias/wrapper for `persona_extraction/extract_vector.py`, kept to make
the quick path explicit as the "prompt extraction" route (single-script mode).
"""

from persona_extraction.extract_vector import main


if __name__ == "__main__":
    main()

