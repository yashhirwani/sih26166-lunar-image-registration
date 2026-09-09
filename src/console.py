"""
console.py
----------
Windows terminals often default stdout/stderr to a legacy codepage
(cp1252/cp437) instead of UTF-8. Printing a Unicode symbol (checkmarks,
warning signs, arrows) then raises UnicodeEncodeError and kills the run
mid-pipeline. Call setup_utf8_console() once, as early as possible, to
make console output resilient regardless of the host codepage.
"""

import sys

_configured = False


def setup_utf8_console() -> None:
    """Reconfigures stdout/stderr to UTF-8 with safe fallback substitution.

    Idempotent — safe to call from every entrypoint (app.py, CLI scripts,
    pipeline.py) without repeating the work.
    """
    global _configured
    if _configured:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream doesn't support reconfigure (e.g. captured/piped in some hosts)
    _configured = True
