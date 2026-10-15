"""window.ml restricted mode: compiling a cell, and running it.

A cell is compiled the way window.ml's python_exec runs it: as the body of a function, so a top-level ``return`` is
legal and is the cell's value. Every code object in the result, the cell's own and each nested ``def``, ``lambda`` and
comprehension, carries ``CO_RESTRICTED``, so a callback the cell hands to a C leaf is recognised as restricted when the
leaf calls it back. Membership (the AST whitelist) is checked before compiling, in a later slice.
"""

import textwrap
import _restricted

__all__ = ["compile_cell", "run_cell", "mark", "CO_RESTRICTED"]

CO_RESTRICTED = _restricted.CO_RESTRICTED


def mark(code):
    """Return ``code`` with CO_RESTRICTED set on it and, recursively, on every code object among its constants."""
    consts = tuple(mark(c) if isinstance(c, type(code)) else c for c in code.co_consts)
    return code.replace(co_flags=code.co_flags | CO_RESTRICTED, co_consts=consts)


def compile_cell(source, filename="<cell>"):
    """Compile a cell's source as the body of ``_user()`` and return a module code object that defines and calls it,
    leaving the return value in the global ``__cell_result__``. Every code object in it is marked restricted."""
    wrapped = "def _user():\n" + textwrap.indent(source, "    ") + "\n__cell_result__ = _user()\n"
    return mark(compile(wrapped, filename, "exec", dont_inherit=True))


def run_cell(source, namespace):
    """Compile and run a cell in ``namespace`` (a dict) with the thread restricted; return the cell's value."""
    _restricted.run(compile_cell(source), namespace)
    return namespace.pop("__cell_result__", None)
