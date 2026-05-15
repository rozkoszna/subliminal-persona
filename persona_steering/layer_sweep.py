#!/usr/bin/env python3
"""
Backward-compatible wrapper.

`layer_sweep.py` was renamed to `steer_sweep.py`.
Use `python persona_steering/steer_sweep.py ...` for new functionality.
"""

from persona_steering.steer_sweep import main


if __name__ == "__main__":
    main()
