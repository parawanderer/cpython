"""window.ml restricted mode: compiling a cell, and running it.

A cell is compiled the way window.ml's python_exec runs it: as the body of a function, so a top-level ``return`` is
legal and is the cell's value. Every code object in the result, the cell's own and each nested ``def``, ``lambda`` and
comprehension, carries ``CO_RESTRICTED``, so a callback the cell hands to a C leaf is recognised as restricted when the
leaf calls it back. Membership (the AST whitelist) is checked before compiling, in a later slice.
"""

import textwrap
import _restricted

__all__ = ["compile_cell", "run_cell", "mark", "load_manifests", "CO_RESTRICTED"]

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


# Leaves that resolve a NAME they are given, or call a callable they are given, need an entry check before they may be
# lifted (repl-specs 2026-10-15: a closed vocabulary for strings, only the program's own functions as callbacks). Until
# that check exists they are registered UNLIFTED: gated, so they fail closed on pandas' own internals.
_NEEDS_ENTRY_CHECK = {"agg", "aggregate", "apply", "transform", "map", "applymap", "pipe", "query", "eval",
                      "sort_values", "groupby"}
# Effects outside heap(owned): never lifted, never listed, until the manifest's effects pass says how.
_EFFECTFUL = {"read_csv", "read_html", "read_json", "read_excel", "read_parquet", "read_pickle", "to_pickle",
              "to_csv", "to_parquet", "to_excel", "to_json", "to_html", "to_sql", "info"}


def load_manifests(data_dir):
    """Register the gate's allow tables from window-ml's notebooks/python-dialect/data: whitelist.json and the
    manifest-*.json name lists. Returns what it registered, for a runner to report. A baseline for running the
    dialect vectors, not the final policy: effects, return marking and costs are the manifests' second pass."""
    import builtins, json, os
    data = lambda name: json.load(open(os.path.join(data_dir, name)))
    wl = data("whitelist.json")
    done = {"calls": [], "lifted": [], "kinds": {}, "modules": {}}

    for name in wl["builtins"]:
        f = getattr(builtins, name, None)
        if f is not None and name not in wl["refused_builtins"]:
            _restricted.allow_call(f, False)
            done["calls"].append(name)
    containers = {"str": str, "list": list, "dict": dict, "set": set, "tuple": tuple, "int": int, "float": float}
    for kind, names in wl["builtin_kinds"].items():
        names = list(names)
        if kind in ("str", "list", "dict", "tuple"):
            names.append("[]")
        if kind in ("list", "dict"):
            names.append("[]=")     # ownership (a later slice) narrows this to owned values
        if kind == "list":
            names.append("sort")    # sorted() looks it up, unlifted
        _restricted.allow_kind(containers[kind], names)
        done["kinds"][kind] = names
    _restricted.allow_kind(range, ["[]"])

    stdlib = data("manifest-stdlib.json")
    for mod, names in stdlib["modules"].items():
        _restricted.allow_module(mod, names)
        done["modules"][mod] = names
    for mod in wl["modules"]:
        if mod not in done["modules"] and mod not in ("pandas", "numpy", "PIL"):
            _restricted.allow_module(mod, [])
            done["modules"][mod] = []

    for mod_name in ("numpy", "pandas"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        man = data(f"manifest-{mod_name}.json")
        top = sorted({f.split(".")[0] for f in man["functions"]} - _EFFECTFUL)
        _restricted.allow_module(mod_name, top)
        done["modules"][mod_name] = top
        for fname in top:
            f = getattr(mod, fname, None)
            if f is not None and fname not in _NEEDS_ENTRY_CHECK:
                _restricted.allow_call(f, True)
                done["lifted"].append(f"{mod_name}.{fname}")
        for kind, names in man["kinds"].items():
            cls = _kind_class(mod_name, kind)
            if cls is None:
                continue
            listed = [n for n in names if n not in _EFFECTFUL]
            extra = ["[]", "[]^"] if (mod_name, kind) in (("pandas", "DataFrame"), ("pandas", "Series")) else []
            for c in (cls if isinstance(cls, tuple) else (cls,)):
                _restricted.allow_kind(c, listed + extra)
                for n in listed:
                    if n in _NEEDS_ENTRY_CHECK:
                        continue
                    f = _static_function(c, n)
                    if f is not None:
                        _restricted.allow_call(f, True)
                        done["lifted"].append(f"{c.__name__}.{n}")
            done["kinds"][f"{mod_name}.{kind}"] = listed + extra
    return done


def _kind_class(mod_name, kind):
    if mod_name == "numpy":
        import numpy
        return {"ndarray": numpy.ndarray}.get(kind)
    import pandas
    from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
    return {"DataFrame": pandas.DataFrame, "Series": pandas.Series, "Index": pandas.Index,
            "GroupBy": (DataFrameGroupBy, SeriesGroupBy)}.get(kind)


def _static_function(cls, name):
    """The plain function a method of `cls` is, found without running a descriptor (a lifted leaf is registered by
    its function, since each bound method is a fresh object)."""
    import inspect
    try:
        f = inspect.getattr_static(cls, name)
    except AttributeError:
        return None
    return f if inspect.isfunction(f) else None
