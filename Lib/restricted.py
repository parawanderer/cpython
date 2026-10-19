"""window.ml restricted mode: compiling a cell, and running it.

A cell is compiled the way window.ml's python_exec runs it: as the body of a function, so a top-level ``return`` is
legal and is the cell's value. Every code object in the result, the cell's own and each nested ``def``, ``lambda`` and
comprehension, carries ``CO_RESTRICTED``, so a callback the cell hands to a C leaf is recognised as restricted when the
leaf calls it back. Membership is checked BEFORE compiling: every node must be in the whitelist (window-ml's
notebooks/python-dialect/data/whitelist.json, the one ratchet), a generator expression is admitted only as the sole
positional argument of a listed consumer and is rewritten to a list comprehension there, a ``print(...)`` call is
rewritten to convert its arguments with the cell's own ``str()`` (print_call), and a dunder name or attribute is
refused outright. A refusal here is NotInDialect before anything runs.
"""

import ast
import textwrap
import _restricted

NotInDialect = _restricted.NotInDialect

__all__ = ["compile_cell", "run_cell", "mark", "load_manifests", "allow_print", "CO_RESTRICTED", "DEFAULT_BUDGET",
           "DEFAULT_MEMORY", "MAX_COLLECTION"]

CO_RESTRICTED = _restricted.CO_RESTRICTED
# Steps a run gets when the caller names none. A step is one periodic check (a backward jump or a call), so it is far
# coarser than the JS dialect's AST-node step; the number is calibrated to the same TIME, about a second of interpreter
# work (Lib/test/test_restricted.py, BudgetTest, measures the rate on this build).
DEFAULT_BUDGET = _restricted.DEFAULT_BUDGET
# Bytes a run may add to the heap, net: the native form of the JS dialect's size caps (MAX_STRING, MAX_COLLECTION).
DEFAULT_MEMORY = _restricted.DEFAULT_MEMORY
# The longest range a program may build: a C consumer (sum, max, sorted, list) walks it with no step the budget can
# see, so its length is what bounds that call's cost (the JS dialect's MAX_COLLECTION).
MAX_COLLECTION = 1_000_000


def mark(code):
    """Return ``code`` with CO_RESTRICTED set on it and, recursively, on every code object among its constants."""
    consts = tuple(mark(c) if isinstance(c, type(code)) else c for c in code.co_consts)
    return code.replace(co_flags=code.co_flags | CO_RESTRICTED, co_consts=consts)


# The node whitelist when none is given: window-ml's ruled set (whitelist.json "nodes", 2026-10-15). A runner passes the
# file's list explicitly; this copy only keeps the module usable on its own.
DEFAULT_NODES = frozenset("""Add And AnnAssign Assert Assign Attribute AugAssign BinOp BitAnd BitOr BitXor BoolOp Break Call
    Compare Constant Continue Del Delete Dict DictComp Div Eq Expr FloorDiv For FormattedValue FunctionDef Gt GtE If
    IfExp Import ImportFrom In Invert Is IsNot JoinedStr LShift Lambda List ListComp Load Lt LtE MatMult Mod Mult Name
    Not NotEq NotIn Or Pass Pow RShift Raise Return Set SetComp Slice Starred Store Sub Subscript Tuple UAdd USub
    UnaryOp alias arg arguments comprehension keyword""".split())
CONSUMERS = frozenset({"sum", "any", "all", "min", "max", "sorted", "list", "tuple", "set", "dict", "enumerate", "zip"})


def _dunder(name):
    return len(name) >= 4 and name.startswith("__") and name.endswith("__")


class _Membership(ast.NodeTransformer):
    """Refuses a node outside the whitelist and a dunder; rewrites a consumer-position genexp to a list comprehension.
    Runs over the CELL's own tree, before the wrapper exists."""

    def __init__(self, nodes):
        self.nodes = nodes

    def refuse(self, node, why):
        raise NotInDialect(f"not in dialect: {why} (line {getattr(node, 'lineno', '?')})")

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id == "print":
            return self.print_call(node)
        consumer = (isinstance(f, ast.Name) and f.id in CONSUMERS) or (isinstance(f, ast.Attribute) and f.attr == "join")
        if consumer and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            g = node.args[0]
            node.args[0] = ast.copy_location(ast.ListComp(elt=g.elt, generators=g.generators), g)
        return self.generic_visit(node)

    def print_call(self, node):
        """``print(a, *b, sep=s)`` becomes ``__cell_print__(str(a), *[str(x) for x in b], sep=s)``: each argument is
        converted by the cell's own (gated) ``str()``, so print admits exactly what ``str()`` admits, and the lifted
        writer only ever handles plain strings. ``file=`` and ``**`` are refused (the only stream is the host's)."""
        for k in node.keywords:
            if k.arg is None:
                self.refuse(node, "print(**...)")
            if k.arg not in ("sep", "end", "flush"):
                self.refuse(node, f"print({k.arg}=...)")
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        to_str = lambda e: ast.copy_location(ast.Call(ast.Name("str", ast.Load()), [e], []), e)
        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                x = ast.Name("__x__", ast.Load())
                comp = ast.ListComp(elt=to_str(x), generators=[
                    ast.comprehension(ast.Name("__x__", ast.Store()), a.value, [], 0)])
                args.append(ast.copy_location(ast.Starred(ast.copy_location(comp, a), ast.Load()), a))
            else:
                args.append(to_str(a))
        return ast.copy_location(ast.Call(ast.Name("__cell_print__", ast.Load()), args, node.keywords), node)

    def generic_visit(self, node):
        kind = type(node).__name__
        if isinstance(node, ast.Name) and node.id == "print":
            self.refuse(node, "print other than called directly")
        if kind == "GeneratorExp":
            self.refuse(node, "a generator expression outside consumer position")
        if kind not in self.nodes:
            self.refuse(node, f"{kind}")
        if isinstance(node, ast.Name) and _dunder(node.id):
            self.refuse(node, f"the name {node.id}")
        if isinstance(node, ast.Attribute) and _dunder(node.attr):
            self.refuse(node, f"the attribute {node.attr}")
        if isinstance(node, (ast.FunctionDef, ast.arg, ast.alias)):
            name = getattr(node, "name", None) or getattr(node, "arg", None) or getattr(node, "asname", None)
            if name and _dunder(name):
                self.refuse(node, f"the name {name}")
        return super().generic_visit(node)


def check(source, nodes=DEFAULT_NODES):
    """The cell's statements, membership-checked and rewritten, as an ``ast`` list for the wrapper's body. Raises
    NotInDialect (or SyntaxError) before anything runs."""
    body = "def _user():\n" + textwrap.indent(source, "    ")
    tree = ast.parse(body)
    stmts = tree.body[0].body
    m = _Membership(frozenset(nodes))
    return [m.visit(s) for s in stmts]


def compile_cell(source, filename="<cell>", nodes=DEFAULT_NODES):
    """Check a cell's membership, then compile it as the body of ``_user()`` (a top-level ``return`` is its value) in
    a module that calls it and leaves the value in ``__cell_result__``. Names the cell binds at its top level are
    declared ``global`` by the WRAPPER, as window.ml's runtime does, so they persist in the namespace; the cell itself
    may not say ``global``. Every code object is marked restricted."""
    stmts = check(source, nodes)
    top = sorted({t.id for s in stmts for t in _top_targets(s)})
    fn = ast.FunctionDef(name="_user", args=ast.arguments(), body=([ast.Global(names=top)] if top else []) + stmts,
                         decorator_list=[], returns=None, type_params=[])
    call = ast.Assign(targets=[ast.Name("__cell_result__", ast.Store())], value=ast.Call(ast.Name("_user", ast.Load()), [], []))
    mod = ast.fix_missing_locations(ast.Module(body=[fn, call], type_ignores=[]))
    return mark(compile(mod, filename, "exec", dont_inherit=True))


def _top_targets(stmt):
    """The names a top-level statement binds: assignment targets, a def's name, an import's binding, a for target."""
    if isinstance(stmt, ast.FunctionDef):
        return [ast.Name(stmt.name)]
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return [ast.Name(a.asname or a.name.split(".")[0]) for a in stmt.names]
    targets = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign, ast.For)):
        targets = [stmt.target]
    out = []
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.append(n)
    return out


def run_cell(source, namespace, nodes=DEFAULT_NODES, budget=None, depth=None, memory=None):
    """Check, compile and run a cell in ``namespace`` (a dict) with the thread restricted; return the cell's value.
    ``budget`` is the step budget (DEFAULT_BUDGET when None), ``depth`` the deepest the cell's own frames may nest,
    ``memory`` the bytes the cell may add to the heap (DEFAULT_MEMORY). What the cell printed before a refusal has
    already reached ``sys.stdout``: the undo log restores values, not output, so the host drops or marks it."""
    code = compile_cell(source, nodes=nodes)
    namespace["__cell_print__"] = cell_print
    try:
        _restricted.run(code, namespace, budget, depth, memory)
    finally:
        namespace.pop("__cell_print__", None)
    namespace.pop("_user", None)
    return namespace.pop("__cell_result__", None)


def cell_print(*texts, sep=None, end=None, flush=False):
    """The writer a cell's ``print(...)`` is compiled to call (``__cell_print__`` in its namespace): plain strings,
    already converted under the gate, written to ``sys.stdout``. Lifted behind _print_check (allow_print)."""
    print(*texts, sep=sep, end=end, flush=flush)


def _print_check(func, self, args, kwargs):
    if not all(type(t) is str for t in args):
        raise NotInDialect("not in dialect: print() of a value str() did not turn into a str")
    kwargs = kwargs or {}
    if (any(type(kwargs.get(k)) not in (str, type(None)) for k in ("sep", "end"))
            or type(kwargs.get("flush", False)) is not bool):
        raise NotInDialect("not in dialect: print()'s sep and end must be str or None, flush a bool")


def allow_print():
    """Let cells print: list the writer a cell's ``print(...)`` compiles to, lifted behind its check."""
    _restricted.allow_call(cell_print, True, _print_check)


# Leaves that resolve a NAME they are given, or call a callable they are given: lifted only behind an ENTRY CHECK
# (repl-specs 2026-10-15): every callable argument, walked into plain containers, must be the program's own restricted
# function, and every string in a names position must be in the closed vocabulary. query and eval are out entirely.
_ENTRY_CHECKED = {"agg", "aggregate", "apply", "transform", "map", "applymap", "pipe", "sort_values", "groupby"}
_NAME_POSITION = {"agg", "aggregate", "apply", "transform"}     # resolve a method name from a string
_OUT = {"query", "eval"}
# The closed vocabulary, starting at the pandas reductions (ruled); the corpus passes only "sum".
VOCABULARY = frozenset({"sum", "mean", "median", "min", "max", "count", "std", "var", "prod", "first", "last",
                        "nunique", "size", "any", "all"})
# Pure builtins every manifest starts from, beyond what the corpus count observed: none resolves a name from its
# arguments (getattr, hasattr, eval and the like stay out, ruled).
PURE_BUILTINS = ("abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
                 "int", "isinstance", "len", "list", "map", "max", "min", "ord", "pow", "print", "range", "repr",
                 "reversed", "round", "set", "sorted", "str", "sum", "tuple", "type", "zip")


def _plain(x):
    return type(x) in (list, tuple, dict, set, frozenset)


def _walk(x):
    """x and, recursively, the elements of plain containers (never into an object such as a frame)."""
    yield x
    if type(x) is dict:
        for v in x.values():
            yield from _walk(v)
    elif _plain(x):
        for v in x:
            yield from _walk(v)


def own_function(f):
    """The program's own function: a plain Python function whose code is CO_RESTRICTED (a lambda or def in a cell)."""
    import types
    return type(f) is types.FunctionType and bool(f.__code__.co_flags & CO_RESTRICTED)


def entry_check(name, method, arguments=True):
    """The checker for a lifted leaf named `name`: run before the leaf, with its arguments. For a METHOD the gate may
    see the plain function with the receiver as the first argument (a method call's LOAD_ATTR hands CALL the unbound
    function), so the receiver is split off before the arguments are judged. ``arguments``: judge the callables and
    the names (the leaves in _ENTRY_CHECKED). A method also gets the WRITE check: ``inplace=True``, or a method that
    writes its receiver (_MUTATES_SELF), needs a receiver the cell owns."""
    names_position = name in _NAME_POSITION
    writes_self = name in _MUTATES_SELF

    def check(func, self, args, kwargs):
        if method:
            if self is None:
                self, args = (args[0] if args else None), args[1:]
            if writes_self or (kwargs or {}).get("inplace"):
                if not _restricted.owned(self):
                    raise NotInDialect(f"not in dialect: {name}() writes a value the cell does not own "
                                       f"({type(self).__name__})")
                _restricted.write_barrier(self)     # logged before the leaf writes it, for a refusal to undo
        if not arguments:
            return
        given = list(args) + list((kwargs or {}).values())
        for i, a in enumerate(given):
            for x in _walk(a):
                if callable(x) and not isinstance(x, type) and not own_function(x):
                    raise NotInDialect(f"not in dialect: {name}() is handed a callable that is not the program's own "
                                       f"function ({type(x).__name__})")
        if names_position:
            head = list(args[:1]) + [v for k, v in (kwargs or {}).items() if k == "func"]
            for a in head:
                for x in _walk(a):
                    if isinstance(x, str) and x not in VOCABULARY:
                        raise NotInDialect(f"not in dialect: {name}({x!r}) names a method outside the vocabulary")
    return check


# OWNERSHIP (slice 5). Methods that write their receiver: read only off a value the cell owns (the gate refuses the
# attribute itself otherwise, so no bound one exists to hand to a C loop).
MUTATORS = {
    list: ["append", "extend", "insert", "pop", "remove", "sort", "reverse", "clear"],
    dict: ["update", "pop", "popitem", "setdefault", "clear"],
    set: ["add", "discard", "remove", "pop", "update", "clear", "difference_update", "intersection_update",
          "symmetric_difference_update"],
    bytearray: ["append", "extend", "insert", "pop", "remove", "reverse", "clear"],
}
# pandas methods that write their receiver whatever their arguments (others only with inplace=True).
_MUTATES_SELF = {"update", "insert", "pop"}
# Lifted leaves whose result is a NEW value, owned by the cell: a STAND-IN until the manifests' second pass carries
# return marking (fresh / alias(i) / internal). Only calls that always build a new object; a reshape, a column, .values
# and every other view stay unowned. The gate also refuses the marking when the result IS the receiver or an argument.
_FRESH = {"numpy": {"array", "zeros", "ones", "arange"}, "pandas": {"DataFrame", "Series", "concat", "to_numeric"},
          ("numpy", "ndarray"): {"copy"}, ("pandas", "DataFrame"): {"copy"}, ("pandas", "Series"): {"copy"}}


# Effects outside heap(owned): never lifted, never listed, until the manifest's effects pass says how.
_EFFECTFUL = {"read_csv", "read_html", "read_json", "read_excel", "read_parquet", "read_pickle", "to_pickle",
              "to_csv", "to_parquet", "to_excel", "to_json", "to_html", "to_sql", "info"}


def _range_cost(func, self, args, kwargs):
    """range's cost check: refused past MAX_COLLECTION elements, before anything walks it."""
    try:
        n = len(range(*args, **(kwargs or {})))
    except OverflowError:
        n = MAX_COLLECTION + 1
    if n > MAX_COLLECTION:
        raise NotInDialect(f"not in dialect: a range of more than {MAX_COLLECTION} elements")


def risky_regex(source):
    """Why a regex could backtrack exponentially, or None: a REPEATED group that itself contains a quantifier or an
    alternation ((a+)+, (\\w+\\s?)*, (a|a)+). sre is C with no match timeout, so one match of such a pattern on a
    40-character string runs for hours in a single call the budget never sees. Conservative by design, as the JS
    dialect's riskyRegex (src/readonly-exec/limits.ts), of which this is a port: (?:x|y)+ is refused with the rest."""
    import re
    stack = []              # one frame per open group: [has a quantifier inside, has an alternation inside]

    def is_repeat(i):
        if i >= len(source):
            return False
        c = source[i]
        if c in "*+":
            return True
        if c != "{":
            return False
        m = re.match(r"\{(\d+)(,(\d*))?\}", source[i:])
        return bool(m) and ((m.group(3) == "" or int(m.group(3)) > 1) if m.group(2) is not None
                            else int(m.group(1)) > 1)

    def mark(k):
        for f in stack:
            f[k] = True

    i = 0
    while i < len(source):
        c = source[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":                        # a class is one atom: skip to its close
            i += 1
            while i < len(source) and source[i] != "]":
                i += 2 if source[i] == "\\" else 1
            i += 1
            continue
        if c == "(":
            stack.append([False, False])
            if source[i + 1:i + 2] == "?":  # a group's own prefix is syntax, not a quantifier
                if source[i + 2:i + 3] == "<" and source[i + 3:i + 4] not in ("=", "!"):
                    i = source.find(">", i)
                    if i < 0:
                        return None
                else:
                    i += 3 if source[i + 2:i + 3] == "<" else 2
            i += 1
            continue
        if c == "|":
            mark(1)
            i += 1
            continue
        if c == ")":
            g = stack.pop() if stack else None
            if g is not None and is_repeat(i + 1) and (g[0] or g[1]):
                return ("a repeated group that contains a quantifier" if g[0]
                        else "a repeated group that contains an alternation")
            if g is not None and is_repeat(i + 1):
                mark(0)
            i += 1
            continue
        if is_repeat(i) or c == "?":
            mark(0)
        i += 1
    return None


# re's functions: lifted leaves (sre is C; re's own Python resolves nothing from its arguments) behind a check that the
# pattern cannot backtrack exponentially, and the callback rule for a callable `repl`.
_REGEX = ("compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split")


def regex_check(name):
    arguments = entry_check(name, False)

    def check(func, self, args, kwargs):
        pattern = args[0] if args else (kwargs or {}).get("pattern")
        if isinstance(pattern, bytes):
            pattern = pattern.decode("latin-1")
        if not isinstance(pattern, str):
            raise NotInDialect(f"not in dialect: re.{name}() needs a pattern string")
        why = risky_regex(pattern)
        if why:
            raise NotInDialect(f"not in dialect: re.{name}({pattern!r}) could backtrack exponentially ({why})")
        arguments(func, self, args, kwargs)
    return check


def load_manifests(data_dir):
    """Register the gate's allow tables from window-ml's notebooks/python-dialect/data: whitelist.json and the
    manifest-*.json name lists. Returns what it registered, for a runner to report. A baseline for running the
    dialect vectors, not the final policy: effects, return marking and costs are the manifests' second pass."""
    import builtins, json, os
    data = lambda name: json.load(open(os.path.join(data_dir, name)))
    wl = data("whitelist.json")
    done = {"calls": [], "lifted": [], "kinds": {}, "modules": {}}

    for name in sorted(set(wl["builtins"]) | set(PURE_BUILTINS)):
        f = getattr(builtins, name, None)
        if f is not None and name not in wl["refused_builtins"]:
            _restricted.allow_call(f, False, _range_cost if f is range else None)
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
    allow_print()
    for t, names in MUTATORS.items():
        _restricted.allow_mutators(t, names)

    stdlib = data("manifest-stdlib.json")
    for mod, names in stdlib["modules"].items():
        _restricted.allow_module(mod, names)
        done["modules"][mod] = names
    import re
    for name in stdlib["modules"].get("re", []):
        if name in _REGEX:
            _restricted.allow_call(getattr(re, name), True, regex_check(name))
            done["lifted"].append(f"re.{name}")
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
            if f is not None and fname not in _OUT:
                _restricted.allow_call(f, True, entry_check(fname, False) if fname in _ENTRY_CHECKED else None)
                if fname in _FRESH.get(mod_name, ()):
                    _restricted.allow_fresh(f)
                done["lifted"].append(f"{mod_name}.{fname}")
        for kind, names in man["kinds"].items():
            cls = _kind_class(mod_name, kind)
            if cls is None:
                continue
            listed = [n for n in names if n not in _EFFECTFUL and n not in _OUT]
            if mod_name == "pandas":    # the checked leaves are listed whatever the corpus count's threshold said
                listed += [n for n in sorted(_ENTRY_CHECKED) if n not in listed]
            # Item writes are listed, and the gate admits them only on an OWNED value (a copy, never the injected
            # frame or a view of it).
            extra = (["[]", "[]=", "[]^"] if (mod_name, kind) in (("pandas", "DataFrame"), ("pandas", "Series"))
                     else ["[]", "[]="] if (mod_name, kind) == ("numpy", "ndarray") else [])
            if mod_name == "pandas":
                # pandas' dunders (__len__, __add__, __eq__, __iter__) are Python: reached through a type slot they run
                # lifted, as its listed methods do, after the gate's own barrier for the operation.
                extra.append("__^")
            for c in (cls if isinstance(cls, tuple) else (cls,)):
                present = [n for n in listed if hasattr(c, n)]
                _restricted.allow_kind(c, present + extra)
                _restricted.allow_mutators(c, [n for n in present if n in _MUTATES_SELF])
                for n in present:
                    f = _static_function(c, n)
                    if f is not None:
                        _restricted.allow_call(f, True, entry_check(n, True, arguments=n in _ENTRY_CHECKED))
                        if n in _FRESH.get((mod_name, kind), ()):
                            _restricted.allow_fresh(f)
                        done["lifted"].append(f"{c.__name__}.{n}")
            if (mod_name, kind) == ("pandas", "DataFrame"):
                # dynamic: columns (ruled). Ours, not pandas' __getattr__; never a name the class itself has.
                _restricted.allow_dynamic(cls, lambda obj, name, _cls=cls: isinstance(name, str)
                                          and not hasattr(_cls, name) and name in obj.columns)
            done["kinds"][f"{mod_name}.{kind}"] = listed + extra
    _register_snapshots()
    return done


def _snapshot_frame(obj):
    return obj.copy(deep=True)


def _restore_frame(obj, state):
    obj._update_inplace(state)      # pandas' own in-place replacement: data, index and columns


def _snapshot_array(obj):
    return obj.copy()


def _restore_array(obj, state):
    import numpy
    numpy.copyto(obj, state)        # same shape and dtype: a write never resizes an ndarray


def _register_snapshots():
    """The undo log's copy and restore for owned values of library kinds (a copy an earlier cell made)."""
    try:
        import numpy
        _restricted.allow_snapshot(numpy.ndarray, _snapshot_array, _restore_array)
    except ImportError:
        pass
    try:
        import pandas
        for cls in (pandas.DataFrame, pandas.Series):
            _restricted.allow_snapshot(cls, _snapshot_frame, _restore_frame)
    except ImportError:
        pass


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
