"""Chainlit entry point.

    uv run chainlit run app.py -w

Chainlit executes its target as a top-level module rather than importing it as part of a
package, so the app lives in `watsonville_motors.ui` and this file just pulls it in. The
Chainlit decorators register on import, which is all the framework needs.
"""

from watsonville_motors.ui import *
