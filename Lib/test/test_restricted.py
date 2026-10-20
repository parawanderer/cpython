"""Tests for window.ml's restricted mode, slice 1: the flag, the entry, and the dynamic extent.

The gate itself (refusing attributes, items, calls and imports) is a later slice; here ``_restricted.active()`` is
the observable, called from inside the code under test. The dialect vectors (window-ml's
tests/fixtures/dialect-vectors/) are the acceptance suite once the gate exists.
"""

import gc
import sysconfig
import types
import unittest

import _restricted
import restricted


_active = _restricted.active


def host_active():
    """Host Python: not restricted code itself. Calls a probe bound in advance: reading `_restricted.active` off the
    module would itself be refused when this runs under the gate (tier 3), which is the point."""
    return _active()


def via_python(f):
    """Host Python that calls the function it is given INLINE (no C entry between them)."""
    return lambda x: f(x)


class BuildTest(unittest.TestCase):
    # Ruled first: the gate is in the abstract layer, which a specialized instruction bypasses, so a runner must
    # refuse to trust anything on an interpreter that specializes, has tier 2 or a JIT.
    def test_00_specialization_tier2_and_jit_are_off(self):
        self.assertEqual(_restricted.build(),
                         {"specialization": False, "specialization_ft": False, "tier2": False, "jit": False})

    def test_a_hot_attribute_read_is_not_specialized(self):
        import dis

        class A:
            x = 1

        def f(o):
            return o.x

        a = A()
        for _ in range(10_000):
            f(a)
        ops = {i.opname for i in dis.get_instructions(f, adaptive=True)}
        self.assertIn("LOAD_ATTR", ops)
        self.assertFalse(any(op.startswith("LOAD_ATTR_") for op in ops), ops)


class CompileTest(unittest.TestCase):
    def setUp(self):
        for f in (sum, max, range):
            _restricted.allow_call(f, False)
        _restricted.allow_module("math", [])

    def tearDown(self):
        _restricted.reset()

    def test_every_nested_code_object_is_marked(self):
        code = restricted.compile_cell("def f():\n    return [x for x in (lambda: [1])()]\nreturn f()")
        seen = []

        def walk(c):
            seen.append(c.co_name)
            self.assertTrue(c.co_flags & restricted.CO_RESTRICTED, c.co_name)
            for k in c.co_consts:
                if isinstance(k, type(c)):
                    walk(k)

        walk(code)
        self.assertIn("_user", seen)
        self.assertIn("<lambda>", seen)

    def test_run_refuses_unmarked_code(self):
        with self.assertRaises(ValueError):
            _restricted.run(compile("1", "<x>", "exec"), {})

    def test_membership_refuses_before_running(self):
        for src in ("while True:\n    pass", "class A:\n    pass", "global x\nx = 1", "with x:\n    pass",
                    "g = (x for x in [1])", "return (x for x in [1])", "return max((x for x in [1]), 5)",
                    "return __builtins__", "return (1).__class__", "def __init__():\n    pass", "yield 1"):
            with self.assertRaises(restricted.NotInDialect, msg=src):
                restricted.compile_cell(src)

    def test_a_consumer_genexp_is_rewritten_to_a_list_comprehension(self):
        code = restricted.compile_cell("return sum(x * x for x in range(4))")
        inner = [c for c in code.co_consts if isinstance(c, type(code))][0]
        self.assertNotIn("<genexpr>", [k.co_name for k in inner.co_consts if isinstance(k, type(code))])
        self.assertEqual(restricted.run_cell("return sum(x * x for x in range(4))", {}), 14)
        self.assertEqual(restricted.run_cell("return max((x for x in [3, 1]), default=0)", {}), 3)

    def test_top_level_names_persist_in_the_namespace(self):
        ns = {}
        restricted.run_cell("import math\nxs = [1]\nfor i in [2]:\n    pass\ndef f():\n    return 1", ns)
        self.assertEqual(sorted(k for k in ns if not k.startswith("_")), ["f", "i", "math", "xs"])

    def test_a_top_level_return_is_the_value(self):
        self.assertEqual(restricted.run_cell("return 6 * 7", {}), 42)

    def test_lines_are_the_cells_own(self):
        # The cell is parsed under the wrapper's `def _user():` line; every line it reports is the cell's, not one past.
        with self.assertRaisesRegex(restricted.NotInDialect, r"While \(line 2\)$"):
            restricted.check("x = 1\nwhile True:\n    pass")
        with self.assertRaises(SyntaxError) as cm:
            restricted.check("x = 1\nx = (")
        self.assertEqual((cm.exception.lineno, cm.exception.offset, cm.exception.text), (2, 5, "x = (\n"))
        # Columns too: a node, and so a traceback's caret, sits where the cell wrote it.
        self.assertEqual([(s.lineno, s.col_offset) for s in restricted.check("x = 1\nif x:\n    y = 2")[1].body], [(3, 4)])
        # A multi-line string keeps its text: nothing is indented into its continuation lines.
        self.assertEqual(restricted.run_cell('s = """a\nb"""\nreturn s', {}), "a\nb")
        try:
            restricted.run_cell("x = 1\ny = 2\nimport os", {})
        except restricted.NotInDialect as e:   # not assertRaises: it drops the traceback
            tb, lines = e.__traceback__, []
        else:
            self.fail("import os was not refused")
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<cell>":
                lines.append(tb.tb_lineno)
            tb = tb.tb_next
        self.assertEqual(lines[-1:], [3], "the run-time refusal's frame is on the cell's line")


def allow_probes():
    """What the extent tests call from restricted code: the probe (unlifted, or it would always read False), the
    test-only lift, and the C types a callback goes through."""
    _restricted.allow_call(_restricted.active, False)
    _restricted.allow_call(_restricted.call_lifted, False)
    for t in (list, map):
        _restricted.allow_call(t, False)


class ExtentTest(unittest.TestCase):
    def setUp(self):
        allow_probes()

    def tearDown(self):
        _restricted.reset()

    def ns(self):
        return {"active": _restricted.active, "lift": _restricted.call_lifted,
                "host_active": host_active, "via_python": via_python}

    def test_outside_any_entry_nothing_is_restricted(self):
        self.assertFalse(_restricted.active())

    def test_the_cell_runs_restricted(self):
        self.assertTrue(restricted.run_cell("return active()", self.ns()))
        self.assertFalse(_restricted.active())

    def test_tier_3_host_python_called_by_the_cell_is_restricted(self):
        self.assertTrue(restricted.run_cell("return host_active()", self.ns()))

    def test_a_lifted_leaf_is_not_restricted(self):
        self.assertFalse(restricted.run_cell("return lift(active)", self.ns()))

    def test_host_python_inside_a_lifted_leaf_is_not_restricted(self):
        # The barrier is the cell's own frame: the walk stops there, so the cell above it does not count.
        self.assertFalse(restricted.run_cell("return lift(host_active)", self.ns()))

    def test_case_2_a_callback_from_a_lifted_c_leaf_is_restricted_again(self):
        self.assertEqual(restricted.run_cell("return lift(list, map(lambda x: active(), [1, 2]))", self.ns()),
                         [True, True])

    def test_case_3_a_callback_reached_inline_through_host_python_is_restricted_again(self):
        self.assertEqual(
            restricted.run_cell("return lift(list, map(via_python(lambda x: active()), [1]))", self.ns()), [True])

    def test_nested_lifts_restore_in_order(self):
        src = "return [lift(lambda: lift(active)), lift(lambda: active()), active()]"
        self.assertEqual(restricted.run_cell(src, self.ns()), [False, True, True])

    def test_an_exception_restores_the_state(self):
        with self.assertRaises(ZeroDivisionError):
            restricted.run_cell("return 1 / 0", self.ns())
        self.assertFalse(_restricted.active())
        with self.assertRaises(ZeroDivisionError):
            restricted.run_cell("return lift(lambda: 1 / 0)", self.ns())
        self.assertFalse(_restricted.active())

    def test_a_restricted_def_called_from_outside_any_entry_is_refused(self):
        # Ruled 2026-10-20 (window-ml PYTHON_DIALECT.md, #895), replacing 2026-10-15's "restriction is a property of the
        # ENTRY": a cell's own function runs only inside a run, never ungated outside one. OutsideRunTest has each route.
        ns = self.ns()
        self.assertTrue(restricted.run_cell("def probe():\n    return active()\nreturn probe()", ns))
        with self.assertRaisesRegex(restricted.NotInDialect, r"^not in dialect: probe\(\) was defined in a cell"):
            ns["probe"]()

    def test_runs_nest(self):
        ns = self.ns()
        ns["inner"] = lambda: restricted.run_cell("return active()", self.ns())
        self.assertEqual(restricted.run_cell("return [lift(inner), active()]", ns), [True, True])



class OutsideRunTest(unittest.TestCase):
    """A CO_RESTRICTED frame entered while no run is active is refused (#895), on every route into it: the check is at
    RESUME, which a call, a generator's or coroutine's resumption and exec() of the code all pass through."""

    WHY = r"^not in dialect: %s\(\) was defined in a cell that ran inside the read-only dialect, unasked, and runs only there; define it again in this cell to call it here$"

    def setUp(self):
        _restricted.allow_call(next, False)
        _restricted.allow_call(list, False)

    def tearDown(self):
        _restricted.reset()

    def cell(self, src):
        ns = {}
        restricted.run_cell(src, ns)
        return ns

    def test_a_direct_call(self):
        ns = self.cell("def f(x):\n    return x + 1")
        self.assertEqual(restricted.run_cell("return f(1)", ns), 2, "inside a run it runs")
        with self.assertRaisesRegex(restricted.NotInDialect, self.WHY % "f"):
            ns["f"](1)

    def test_a_lambda_is_named_as_the_cell_wrote_it(self):
        ns = self.cell("g = lambda: open")
        with self.assertRaisesRegex(restricted.NotInDialect, self.WHY % "<lambda>"):
            ns["g"]()

    def test_a_nested_function_it_returns(self):
        ns = self.cell("def outer():\n    return lambda: 1\nh = outer()")
        with self.assertRaisesRegex(restricted.NotInDialect, self.WHY % r"outer\.<locals>\.<lambda>"):
            ns["h"]()

    def test_exec_of_its_code(self):
        ns = self.cell("g = lambda: open")
        with self.assertRaises(restricted.NotInDialect):
            exec(ns["g"].__code__, {})
        with self.assertRaises(restricted.NotInDialect):
            exec(restricted.compile_cell("return 1"), {})

    def test_a_c_library_calling_it_from_host_code(self):
        ns = self.cell("f = lambda x: x")
        for call in (lambda: list(map(ns["f"], [1])), lambda: sorted([2, 1], key=ns["f"])):
            with self.assertRaises(restricted.NotInDialect):
                call()

    def test_a_generator_started_and_resumed_outside(self):
        code = restricted.mark(compile("def gen():\n    yield 1\n    yield 2", "<cell>", "exec"))
        gen = types.FunctionType([c for c in code.co_consts if isinstance(c, types.CodeType)][0], {})
        with self.assertRaises(restricted.NotInDialect):
            next(gen())
        # Started inside a run, resumed outside it: the resumption is refused too.
        ns = {"gen": gen}
        restricted.run_cell("it = gen()\nfirst = next(it)", ns)
        self.assertEqual(ns["first"], 1)
        with self.assertRaises(restricted.NotInDialect):
            next(ns["it"])

    def test_a_coroutine(self):
        code = restricted.mark(compile("async def co():\n    return 3", "<cell>", "exec"))
        co = types.FunctionType([c for c in code.co_consts if isinstance(c, types.CodeType)][0], {})
        c = co()
        with self.assertRaises(restricted.NotInDialect):
            c.send(None)

    def test_host_code_is_untouched(self):
        # Positive control: unmarked code runs as ever, and a run's state is restored around the refusal.
        ns = self.cell("f = lambda: 1")
        with self.assertRaises(restricted.NotInDialect):
            ns["f"]()
        self.assertEqual(list(map(lambda x: x + 1, [1])), [2])
        self.assertFalse(_restricted.active())


def peek(o):
    return o.secret


def peek_cold(o):
    return o.secret


class Box:
    secret = 1


class GateTest(unittest.TestCase):
    """Slice 2: the gate refuses what is not listed, on every path, and lets the listed through."""

    @classmethod
    def setUpClass(cls):
        import operator
        # A warmed library function (the warm-library vector): run hot before any restricted entry.
        for _ in range(10_000):
            peek(Box())

    def setUp(self):
        import math, operator
        allow_probes()
        A = _restricted.allow_kind
        A(str, ["format", "format_map", "upper", "lower", "split", "join", "strip", "replace", "[]"])
        A(list, ["append", "copy", "index", "sort", "[]", "[]="])  # sort: sorted() looks it up, unlifted
        A(dict, ["get", "items", "keys", "values", "update", "[]", "[]="])
        A(tuple, ["[]"])
        A(operator.attrgetter, ["()"])
        for f in (len, print, sorted, sum, min, max, range, list, dict, tuple, str, int, float, isinstance, abs,
                  round, enumerate, zip, getattr, hasattr):
            _restricted.allow_call(f, False)
        _restricted.allow_module("math", ["sqrt", "floor", "pi"])
        _restricted.allow_module("operator", ["attrgetter", "itemgetter"])
        self.math = math

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, **names):
        ns = {"active": _restricted.active, "lift": _restricted.call_lifted}
        ns.update(names)
        return restricted.run_cell(src, ns)

    def refused(self, src, **names):
        with self.assertRaises(_restricted.NotInDialect, msg=src):
            self.run_cell(src, **names)
        self.assertFalse(_restricted.active(), "a refusal restores the thread state")

    # --- the listed runs ---

    def test_the_cells_own_data(self):
        self.assertEqual(self.run_cell("xs = []\nxs.append(1)\nxs[0] = 2\nd = {}\nd['a'] = xs[0]\nreturn d"), {"a": 2})

    def test_a_listed_module_and_attribute(self):
        self.assertEqual(self.run_cell("import math\nreturn math.sqrt(16) + math.floor(1.5)"), 5.0)
        self.assertEqual(self.run_cell("from math import floor\nreturn floor(2.5)"), 2)

    def test_listed_builtins(self):
        self.assertEqual(self.run_cell("return sorted([3, 1, 2], key=abs)"), [1, 2, 3])
        self.assertEqual(self.run_cell("return sum(x * x for x in range(10))"), 285)
        self.assertEqual(self.run_cell("return ', '.join(str(x) for x in [1, 2])"), "1, 2")
        self.assertEqual(self.run_cell("return '{0} and {1}'.format(1, 'b')"), "1 and b")

    # --- dunders, directly and through the C paths ---

    def test_a_dunder_attribute(self):
        self.refused("return (1).__class__")

    def test_the_subclasses_walk(self):
        self.refused("return ().__class__.__base__.__subclasses__()")

    def test_str_format_walks_an_attribute_in_c(self):
        self.refused('return "{0.__class__}".format(1)')

    def test_str_format_map_walks_an_attribute_in_c(self):
        self.refused('return "{x.__class__}".format_map({"x": 1})')

    def test_getattr_with_a_computed_dunder(self):
        self.refused('return getattr(1, "__" + "class__")')

    def test_hasattr_cannot_probe_a_dunder(self):
        self.refused('return hasattr(1, "__class__")')

    def test_attrgetter_reaches_a_dunder_from_c(self):
        self.refused('import operator\nreturn operator.attrgetter("__class__")(1)')

    def test_attrgetter_as_a_sort_key(self):
        self.refused('import operator\nreturn sorted([1, 2], key=operator.attrgetter("__class__"))')

    def test_a_traceback_frame(self):
        self.refused("try:\n    1 / 0\nexcept Exception as e:\n    return e.__traceback__.tb_frame")

    def test_a_generators_frame_without_a_dunder(self):
        self.refused("g = (x for x in [1])\nreturn g.gi_frame")

    def test_an_unlisted_attribute_of_a_listed_kind(self):
        self.refused("return 'a'.encode()")

    # --- calls, imports, writes ---

    def test_an_unlisted_builtin(self):
        self.refused('return eval("1")')
        self.refused('return open("/etc/hosts")')
        self.refused("return type(1)")

    def test_an_unlisted_import(self):
        self.refused("import os\nreturn 1")
        self.refused("import math\nreturn math.__dict__")
        self.refused("from math import tau\nreturn tau")

    def test_a_relative_import(self):
        with self.assertRaises(ImportError):
            self.run_cell("from . import x\nreturn 1")

    def test_setting_an_attribute(self):
        self.refused("import math\nmath.pi = 3\nreturn 1")

    def test_an_item_write_on_an_unlisted_kind(self):
        self.refused("t = (1, 2)\nt[0] = 3\nreturn t")

    # --- the extent under the gate ---

    def test_warm_library_function_is_still_gated(self):
        self.refused("return peek(box)", peek=peek, box=Box())

    def test_cold_library_function_is_gated(self):
        self.refused("return peek_cold(box)", peek_cold=peek_cold, box=Box())

    def test_a_callback_from_a_lifted_leaf_is_gated(self):
        self.refused("return lift(list, map(lambda o: o.secret, [box]))", box=Box())

    def test_a_callback_reached_inline_through_host_python_is_gated(self):
        self.refused("return lift(list, map(via_python(lambda o: o.secret), [box]))", box=Box(), via_python=via_python)

    def test_lift_and_reentry_nest_twice(self):
        # The callback calls a lifted leaf, which calls back again: each restricted frame is gated, each leaf is not.
        src = ("return lift(list, map(lambda x: [active(), lift(list, map(lambda y: [active(), lift(active)], [1]))],"
               " [1]))")
        self.assertEqual(self.run_cell(src), [[True, [[True, False]]]])
        self.refused("return lift(list, map(lambda x: lift(list, map(lambda y: y.secret, [x])), [box]))", box=Box())


def swallow(f, *args):
    """Host code that catches everything, the way pandas' fallbacks do: registered as a LIFTED leaf."""
    try:
        return f(*args)
    except Exception:
        return "swallowed"


def each(f, n):
    """Host code that calls a callback n times (a C leaf's loop, from the budget's side)."""
    for i in range(n):
        f(i)
    return n


class BudgetTest(unittest.TestCase):
    """Slice 4: the step budget at the eval breaker, the depth, a C loop's cost bounded before it runs, and a refusal
    that library code catches still refusing the run."""

    LOOP = "n = 0\nfor i in range(100):\n    n += i\nreturn n"

    def setUp(self):
        for f in (len, sum, max, list, abs):
            _restricted.allow_call(f, False)
        _restricted.allow_call(range, False, restricted._range_cost)
        _restricted.allow_kind(list, ["append", "[]", "[]="])
        _restricted.allow_kind(str, ["format"])
        _restricted.allow_call(swallow, True)
        _restricted.allow_call(each, True)

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, budget=None, depth=None, **names):
        ns = dict(names, swallow=swallow, each=each)
        return restricted.run_cell(src, ns, budget=budget, depth=depth)

    def refused(self, src, match, **kw):
        with self.assertRaisesRegex(_restricted.NotInDialect, match, msg=src):
            self.run_cell(src, **kw)
        self.assertFalse(_restricted.active(), "a refusal restores the thread state")

    # --- the step budget ---

    def test_a_bounded_loop_runs_out(self):
        self.refused(self.LOOP, "step budget", budget=50)
        self.assertEqual(_restricted.steps(), 51)

    def test_the_same_loop_under_enough_budget(self):
        self.assertEqual(self.run_cell(self.LOOP, budget=500), 4950)
        self.assertGreaterEqual(_restricted.steps(), 100)
        self.assertLess(_restricted.steps(), 120)

    def test_a_list_that_grows_while_it_is_iterated(self):
        self.refused("xs = [1]\nfor x in xs:\n    xs.append(x)\nreturn len(xs)", "step budget", budget=200_000)

    def test_a_nested_comprehension(self):
        self.refused("return len([i * j for i in range(10 ** 5) for j in range(10 ** 5)])", "step budget",
                     budget=200_000)

    def test_calls_are_charged(self):
        self.refused("def f(x):\n    return x\nreturn [f(1), f(2), f(3), f(4)]", "step budget", budget=3)

    def test_a_callback_from_a_lifted_leaf_is_charged(self):
        self.refused("return each(lambda i: i, 1000)", "step budget", budget=200)

    def test_no_budget_means_the_default(self):
        self.assertEqual(self.run_cell(self.LOOP), 4950)
        self.assertEqual(restricted.DEFAULT_BUDGET, _restricted.DEFAULT_BUDGET)
        with self.assertRaises(ValueError):
            self.run_cell(self.LOOP, budget=-1)

    def test_the_default_is_about_a_second(self):
        import time
        t = time.perf_counter()
        self.run_cell("n = 0\nfor i in range(1000000):\n    n += i\nreturn n", budget=2_000_000)
        rate = _restricted.steps() / (time.perf_counter() - t)
        seconds = restricted.DEFAULT_BUDGET / rate
        # Broad, for a loaded machine: the point is the order of magnitude (the JS dialect's budget is ~1 s).
        self.assertGreater(seconds, 0.1, f"{rate:.0f} steps/s")
        self.assertLess(seconds, 10, f"{rate:.0f} steps/s")

    def test_the_budget_ends_with_the_run(self):
        f = self.run_cell("def f(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\nreturn f",
                          budget=100)
        # Its run's budget does not follow it out: outside a run it is refused, never metered on a spent budget (#895).
        with self.assertRaisesRegex(restricted.NotInDialect, r"^not in dialect: f\(\) was defined in a cell"):
            f(1000)
        self.refused(self.LOOP, "step budget", budget=50)
        self.assertEqual(self.run_cell(self.LOOP, budget=500), 4950, "an exhausted budget does not leak to the next")

    def test_exhaustion_is_sticky(self):
        # The first refusal is swallowed by host code; the cell loops on, meets the budget again, and is refused.
        self.refused("swallow(lambda: [i for i in range(1000)])\nreturn [i for i in range(1000)]", "step budget",
                     budget=100)

    # --- a refusal library code catches ---

    def test_a_swallowed_budget_refusal_still_refuses(self):
        # Refused at the next periodic check after the swallow (sticky), or by run() at the end: either way, refused.
        self.refused("return swallow(lambda: [i for i in range(1000)])", "step budget|caught by library code",
                     budget=100)

    def test_a_swallowed_gate_refusal_still_refuses(self):
        self.refused("return swallow(lambda: '{0.__class__}'.format(1))", "caught by library code")

    def test_an_unrefused_swallow_is_a_value(self):
        self.assertEqual(self.run_cell("return swallow(lambda: 1 // 0)"), "swallowed")

    # --- the depth ---

    def test_recursion_past_the_depth(self):
        self.refused("def f(n):\n    return f(n + 1)\nreturn f(0)", "depth")

    def test_mutual_recursion_past_the_depth(self):
        self.refused("def a(n):\n    return b(n + 1)\ndef b(n):\n    return a(n + 1)\nreturn a(0)", "depth")

    def test_recursion_within_the_depth(self):
        self.assertEqual(self.run_cell("def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nreturn f(200)"), 200)

    def test_a_smaller_depth(self):
        src = "def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\nreturn f(20)"
        self.refused(src, "depth", depth=10)
        self.assertEqual(self.run_cell(src, depth=30), 20)

    def test_the_refusal_is_not_a_recursion_error(self):
        try:
            self.run_cell("def f(n):\n    return f(n + 1)\nreturn f(0)")
        except _restricted.NotInDialect:
            pass
        except RecursionError:
            self.fail("CPython's own limit was reached before the dialect's depth")

    # --- a C loop's cost, bounded before it runs ---

    def test_a_sum_over_a_huge_range(self):
        self.refused("return sum(range(10 ** 12))", "range of more than")

    def test_a_range_too_big_to_measure(self):
        self.refused("return max(range(10 ** 30))", "range of more than")

    def test_a_range_at_the_cap(self):
        self.assertEqual(self.run_cell(f"return len(range({restricted.MAX_COLLECTION}))"), restricted.MAX_COLLECTION)
        self.refused(f"return len(range(-1, {restricted.MAX_COLLECTION}))", "range of more than")


def make_list():
    """Host code that builds a list: lifted, so the list is NOT the cell's (not fresh-marked)."""
    return [1]


def make_fresh():
    """The same, registered as a FRESH leaf."""
    return [1]


def give_back(x):
    """A leaf registered FRESH that hands back its argument: the gate must not mark that owned."""
    return x


def make_gen():
    """A FRESH leaf returning a generator made in its lifted extent: the generator is in the owned set AND lifted_gens."""
    return (i for i in range(2))


def plain_gen():
    """A lifted leaf that is NOT fresh: its generator is only in lifted_gens."""
    return (i for i in range(2))


def probe_gen():
    """Created under the gate (not fresh, not lifted): its body must resume gated. A stale lifted_gens entry at this
    address would resume it lifted, and `().__class__` would yield instead of being refused."""
    yield ().__class__


class OwnershipTest(unittest.TestCase):
    """Slice 5: writes need a value the cell owns (made under the gate, or a FRESH leaf result), on every path: item
    writes, deletes, in-place operators, and a mutating method, which is not even read off a value the cell does not
    own (so no bound one reaches a C loop)."""

    def setUp(self):
        for f in (len, list, map, sorted, sum, bytearray, set, dict, range):
            _restricted.allow_call(f, False)
        _restricted.allow_kind(list, ["append", "extend", "sort", "copy", "[]", "[]="])
        _restricted.allow_kind(dict, ["get", "update", "[]", "[]="])
        _restricted.allow_kind(set, ["add"])
        _restricted.allow_kind(bytearray, ["append", "[]", "[]="])
        for t, names in restricted.MUTATORS.items():
            _restricted.allow_mutators(t, names)
        for f in (make_list, make_fresh, give_back, each):
            _restricted.allow_call(f, True)
        _restricted.allow_fresh(make_fresh)
        _restricted.allow_fresh(give_back)

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, ns=None, **names):
        ns = {} if ns is None else ns
        ns.update(names, make_list=make_list, make_fresh=make_fresh, give_back=give_back, each=each)
        return restricted.run_cell(src, ns)

    def refused(self, src, ns=None, **names):
        with self.assertRaisesRegex(_restricted.NotInDialect, "does not own", msg=src):
            self.run_cell(src, ns, **names)

    # --- the cell's own values ---

    def test_a_list_the_cell_built(self):
        self.assertEqual(self.run_cell("xs = []\nxs.append(1)\nxs[0] = 2\nxs += [3]\nxs *= 2\ndel xs[0]\n"
                                       "return xs"), [3, 2, 3])

    def test_a_dict_and_a_set_the_cell_built(self):
        self.assertEqual(self.run_cell("d = {}\nd['a'] = 1\nd.update({'b': 2})\nd |= {'c': 3}\ns = set()\n"
                                       "s.add(1)\ns |= {2}\nreturn [d, len(s)]"), [{"a": 1, "b": 2, "c": 3}, 2])

    def test_a_bytearray_the_cell_built(self):
        self.assertEqual(self.run_cell("b = bytearray(2)\nb[0] = 7\nb.append(1)\nreturn len(b)"), 3)

    def test_an_earlier_cells_list(self):
        ns = {}
        self.run_cell("xs = [1]", ns)
        self.assertEqual(self.run_cell("xs.append(2)\nreturn xs", ns), [1, 2])

    def test_sorted_sorts_its_own_new_list(self):
        self.assertEqual(self.run_cell("return sorted(xs)", xs=[3, 1, 2]), [1, 2, 3])

    def test_a_list_built_in_a_callback_from_a_lifted_leaf(self):
        self.assertEqual(self.run_cell("ys = []\neach(lambda i: ys.append(i), 3)\nreturn ys"), [0, 1, 2])

    def test_numbers_and_strings_are_rebound_not_written(self):
        self.assertEqual(self.run_cell("n = 1\nn += 1\nt = 'a'\nt += 'b'\nu = (1,)\nu += (2,)\n"
                                       "return [n, t, u]"), [2, "ab", (1, 2)])

    # --- injected values ---

    def test_an_injected_list(self):
        for src in ("xs.append(1)", "xs[0] = 9", "del xs[0]", "xs += [1]", "xs *= 2", "xs.sort()",
                    "xs.extend([1])", "xs[0:1] = [5]"):
            xs = [3, 1, 2]
            self.refused(src, xs=xs)
            self.assertEqual(xs, [3, 1, 2], src)

    def test_an_injected_dict_and_set(self):
        for src in ("d['b'] = 2", "d.update({'b': 2})", "d |= {'b': 2}", "del d['a']", "s |= {2}", "s.add(2)"):
            d, st = {"a": 1}, {1}
            self.refused(src, d=d, s=st)
            self.assertEqual((d, st), ({"a": 1}, {1}), src)

    def test_reads_of_an_injected_value_still_work(self):
        self.assertEqual(self.run_cell("return [xs[0], d.get('a'), len(xs)]", xs=[1], d={"a": 2}), [1, 2, 1])

    def test_a_bound_mutator_cannot_reach_a_c_loop(self):
        xs = []
        self.refused("return list(map(xs.append, [1, 2]))", xs=xs)
        self.assertEqual(xs, [])

    def test_a_copy_of_an_injected_list_is_the_cells(self):
        xs = [1]
        self.assertEqual(self.run_cell("ys = list(xs)\nys.append(2)\nzs = xs[:]\nzs.append(3)\nreturn [ys, zs]",
                                       xs=xs), [[1, 2], [1, 3]])
        self.assertEqual(xs, [1])

    def test_an_owned_container_of_injected_values(self):
        inner = [1]
        self.refused("outer = [inner]\nouter[0].append(2)", inner=inner)
        self.assertEqual(inner, [1])

    # --- leaf results ---

    def test_a_lifted_leafs_result_is_not_owned(self):
        self.refused("ys = make_list()\nys.append(2)")

    def test_a_fresh_leafs_result_is_owned(self):
        self.assertEqual(self.run_cell("ys = make_fresh()\nys.append(2)\nreturn ys"), [1, 2])

    def test_a_fresh_leaf_handing_back_its_argument(self):
        xs = [1]
        self.refused("give_back(xs).append(2)", xs=xs)
        self.assertEqual(xs, [1])

    # --- the set follows the heap ---

    def test_a_freed_owned_address_is_never_owned_again(self):
        self.run_cell("for i in range(200):\n    t = [i]\nreturn 1")
        made = [[i] for i in range(5000)]     # host lists, some at addresses the cell's freed lists had
        self.assertFalse(any(_restricted.owned(x) for x in made))

    def test_a_cells_list_is_owned_after_the_run(self):
        xs = self.run_cell("return [1]")
        self.assertTrue(_restricted.owned(xs))
        self.assertFalse(_restricted.owned([1]))

    def test_a_generator_in_both_tables_leaves_both(self):
        _restricted.allow_call(make_gen, True)
        _restricted.allow_fresh(make_gen)
        ns = {"make_gen": make_gen}
        self.run_cell("return 1", ns)           # the namespace's own entries exist before `before` is read
        before = _restricted._table_sizes()
        self.run_cell("g = make_gen()", ns)
        self.assertTrue(_restricted.owned(ns["g"]))
        del ns["g"]
        self.assertEqual(_restricted._table_sizes(), before)   # a stale lifted_gens entry resumes a later one lifted


class AddressReuseTest(unittest.TestCase):
    """The address-keyed tables follow the heap. An entry is removed when its object dies (PyRefTracer DESTROY,
    which fires before the freelist or GC can hand the address out again), unconditionally and independently of
    the other tables. A stale owned entry would let a later object at that address be written as the cell's own; a
    stale lifted_gens entry would resume a later generator at that address outside the gate. Both, not either: a
    generator made in a lifted extent and returned by a FRESH leaf sits in both tables and must leave both."""

    def setUp(self):
        for f in (len, list, range, next):
            _restricted.allow_call(f, False)
        _restricted.allow_kind(list, ["append", "[]", "[]="])
        _restricted.allow_call(make_gen, True)
        _restricted.allow_fresh(make_gen)
        _restricted.allow_call(plain_gen, True)
        _restricted.allow_call(probe_gen, False)    # under the gate: its body stays gated

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, ns=None, **names):
        ns = {} if ns is None else ns
        ns.update(names, make_gen=make_gen, plain_gen=plain_gen, probe_gen=probe_gen)
        return restricted.run_cell(src, ns)

    # --- removal is synchronous with death ---

    def test_an_owned_entry_is_gone_at_del_before_anything_reuses_the_address(self):
        xs = self.run_cell("return [1]")
        before = _restricted._table_sizes()
        self.assertTrue(_restricted.owned(xs))
        del xs
        self.assertEqual(_restricted._table_sizes(), (before[0] - 1, before[1]))

    def test_a_lifted_entry_is_gone_at_del_before_anything_reuses_the_address(self):
        ns = {}
        self.run_cell("g = plain_gen()", ns)
        before = _restricted._table_sizes()
        del ns["g"]
        self.assertEqual(_restricted._table_sizes(), (before[0], before[1] - 1))

    # --- cycle collection takes the same path ---

    def test_a_cycle_of_owned_lists_leaves_the_table(self):
        before = _restricted._table_sizes()
        self.run_cell("pairs = []\nfor i in range(50):\n    a = [i]\n    b = [a]\n    a.append(b)\n    pairs.append(a)\n"
                      "pairs = None\nreturn 1")
        gc.collect()
        self.assertEqual(_restricted._table_sizes(), before)
        made = [[i] for i in range(2000)]
        self.assertFalse(any(_restricted.owned(x) for x in made))

    def test_a_generator_in_a_cycle_leaves_both_tables(self):
        ns = {"make_gen": make_gen}
        self.run_cell("return 1", ns)
        before = _restricted._table_sizes()
        self.run_cell("g = make_gen()\nbox = [g]\ng = None\nbox.append(box)\nbox = None\nreturn 1", ns)
        gc.collect()
        self.assertEqual(_restricted._table_sizes(), before)

    def test_a_lifted_only_generator_in_a_cycle_leaves_lifted_gens(self):
        ns = {"plain_gen": plain_gen}
        self.run_cell("return 1", ns)
        before = _restricted._table_sizes()
        self.run_cell("g = plain_gen()\nbox = [g]\ng = None\nbox.append(box)\nbox = None\nreturn 1", ns)
        gc.collect()
        self.assertEqual(_restricted._table_sizes(), before)

    # --- a reused address is never owned or lifted by accident ---

    def test_a_probe_generator_after_a_freed_lifted_one_resumes_gated(self):
        self.run_cell("gs = [plain_gen() for _ in range(300)]\ngs = None\nreturn 1")
        # Every probe is created under the gate and resumed under the gate, one cell each: a stale lifted_gens
        # entry at a reused address would resume that one lifted, and `().__class__` would yield instead of
        # being refused. (A caught refusal inside one cell would trip the run, so each resume is its own cell.)
        made = self.run_cell("return [probe_gen() for _ in range(300)]")
        for g in made:
            with self.assertRaises(_restricted.NotInDialect):
                self.run_cell("return next(g)", {"g": g, "next": next})

    def test_a_probe_generator_after_a_cycle_collected_lifted_one_resumes_gated(self):
        self.run_cell("gs = [plain_gen() for _ in range(200)]\nbox = [gs]\ngs = None\nbox.append(box)\nbox = None\n"
                      "return 1")
        gc.collect()
        made = self.run_cell("return [probe_gen() for _ in range(200)]")
        for g in made:
            with self.assertRaises(_restricted.NotInDialect):
                self.run_cell("return next(g)", {"g": g, "next": next})

    def test_the_tables_do_not_grow_across_many_runs(self):
        self.run_cell("return 1", {})
        before = _restricted._table_sizes()
        for i in range(20):
            self.run_cell("t = [i]\ng = plain_gen()\ndel t, g\nreturn 1", {"i": i})
        self.assertEqual(_restricted._table_sizes(), before)


class MemoryTest(unittest.TestCase):
    """Slice 5: a byte cap per run, counted where CPython takes memory for blocks over 512 bytes and for arenas, net of
    what the run frees. A request past the cap fails before anything is allocated."""

    def setUp(self):
        for f in (len, list, range, sum):
            _restricted.allow_call(f, False, restricted._range_cost if f is range else None)
        _restricted.allow_call(swallow, True)

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, memory=None, **names):
        return restricted.run_cell(src, dict(names, swallow=swallow), memory=memory)

    def refused(self, src, **kw):
        with self.assertRaisesRegex(_restricted.NotInDialect, "memory", msg=src):
            self.run_cell(src, **kw)

    def test_a_string_repeated_past_memory(self):
        import time
        t = time.perf_counter()
        self.refused('return len("a" * 10 ** 11)')
        self.assertLess(time.perf_counter() - t, 1, "refused before the allocation, not after it")

    def test_a_doubling_list(self):
        self.refused("x = [1]\nfor _ in range(64):\n    x = x + x\nreturn len(x)")

    def test_a_list_within_the_cap(self):
        self.assertEqual(self.run_cell("return len([0] * 10 ** 6)"), 10 ** 6)
        self.assertGreaterEqual(_restricted.memory(), 8 * 10 ** 6)

    def test_a_smaller_cap(self):
        self.refused("return len([0] * 10 ** 6)", memory=10 ** 6)

    def test_many_small_objects_count_too(self):
        # 2e5 small lists live at once: pymalloc arenas, not RAW blocks.
        self.refused("return len([[i] for i in range(200000)])", memory=4 * 10 ** 6)

    def test_freed_memory_is_given_back(self):
        # 100 lists of 8 MB, each dropped before the next: about 8 MB live at a time, under a 64 MB cap.
        self.assertEqual(self.run_cell("n = 0\nfor i in range(100):\n    x = [0] * 10 ** 6\n    n += len(x)\n"
                                       "return n", memory=64 * 10 ** 6), 10 ** 8)

    def test_a_swallowed_memory_error_still_refuses(self):
        self.refused('return swallow(lambda: "a" * 10 ** 11)')

    def test_the_cap_ends_with_the_run(self):
        self.refused("return len([0] * 10 ** 6)", memory=10 ** 6)
        self.assertEqual(len("a" * 10 ** 7), 10 ** 7, "the host allocates freely after a run")
        self.assertEqual(self.run_cell("return len([0] * 10 ** 6)"), 10 ** 6, "the next run gets a fresh cap")

    def test_a_repeat_past_py_ssize_t_is_the_cap(self):
        # CPython's own guards refuse these with no allocation (OverflowError, MemoryError), so they never tripped the
        # run. 10 ** 20 is past Py_ssize_t here as 10 ** 11 is on wasm32, where the halting vector met it.
        for src in ('return len("a" * 10 ** 20)', 'return len("ab" * 2 ** 62)', "return len([0, 0] * 2 ** 62)",
                    'return len(b"a" * 10 ** 20)', "return len((1,) * 10 ** 20)", 'x = "a"\nx *= 10 ** 20\nreturn x',
                    'return len(10 ** 20 * "a")'):
            self.refused(src)

    def test_a_block_is_counted_once(self):
        # An 8 MB item array is one block: counted at OBJ/MEM or at RAW, never both. Under pymalloc MEM and OBJ are
        # not wrapped at all (they reach RAW and the arenas), whatever the allocator name reads once RAW is wrapped.
        self.assertEqual(self.run_cell("x = [0] * 10 ** 6\nreturn len(x)"), 10 ** 6)
        self.assertGreaterEqual(_restricted.memory(), 8 * 10 ** 6)
        self.assertLess(_restricted.memory(), 9 * 10 ** 6)
        # The allocator this interpreter started with, not the environment (-E ignores PYTHONMALLOC): NOT_SET, DEFAULT,
        # DEBUG and the two pymalloc ones put pymalloc under MEM and OBJ.
        import _testinternalcapi
        pymalloc = _testinternalcapi.get_configs()["pre_config"]["allocator"] in (0, 1, 2, 5, 6)
        if pymalloc:
            self.assertEqual(_restricted._memory_wrapped()[0], 0, "pymalloc routes MEM and OBJ through RAW and the arenas")
        else:
            self.assertEqual(_restricted._memory_wrapped()[0], 1)

    def test_mem_and_obj_are_wrapped_only_inside_a_run(self):
        seen = []
        def probe():
            seen.append(_restricted._memory_wrapped())
            return 0
        _restricted.allow_call(probe, True)
        self.run_cell("return probe()", probe=probe)
        wraps, mem, obj = seen[0]
        self.assertEqual((mem, obj), (bool(wraps), bool(wraps)), "inside the run")
        # A free-threaded build keeps them installed: swapping them without a GIL would race another thread's allocation.
        after = (bool(wraps), bool(wraps)) if sysconfig.get_config_var("Py_GIL_DISABLED") else (False, False)
        self.assertEqual(_restricted._memory_wrapped()[1:], after, "after it")

    def test_a_hook_stacked_during_a_run_is_kept_and_nothing_is_counted_twice(self):
        import tracemalloc
        if tracemalloc.is_tracing():
            # Its stop() would put back the allocators saved at startup, from before ours, and every later test here
            # would count nothing.
            self.skipTest("tracemalloc was started before the first run")
        def start():
            tracemalloc.start()
            return 0
        _restricted.allow_call(start, True)
        try:
            self.run_cell("return start()", start=start)
            self.assertTrue(tracemalloc.is_tracing(), "the run's end did not drop the hook stacked over ours")
            before = tracemalloc.get_traced_memory()[0]
            x = [0] * 10 ** 5
            self.assertGreater(tracemalloc.get_traced_memory()[0], before, "and it still sees allocations")
            del x
        finally:
            tracemalloc.stop()
        for _ in range(2):      # the first run takes out the wrapper tracemalloc put back; the second wraps afresh
            self.assertEqual(self.run_cell("x = [0] * 10 ** 6\nreturn len(x)"), 10 ** 6)
            self.assertLess(_restricted.memory(), 9 * 10 ** 6, "counted once")
            wraps = bool(_restricted._memory_wrapped()[0])
            gil_free = sysconfig.get_config_var("Py_GIL_DISABLED")
            self.assertEqual(_restricted._memory_wrapped()[1:], (wraps, wraps) if gil_free else (False, False))

    def test_an_empty_or_negative_repeat_is_not_judged(self):
        # Neither asks for memory, so each keeps CPython's own error: past Py_ssize_t is an OverflowError either way.
        self.assertEqual(self.run_cell('return len("a" * -10 ** 6)'), 0)
        for src in ('return len("" * 10 ** 20)', 'return len("a" * -10 ** 20)', "return len([] * 10 ** 20)"):
            with self.assertRaises(OverflowError, msg=src):
                self.run_cell(src)
        with self.assertRaises(OverflowError):
            "a" * 10 ** 20      # the host's repeat is CPython's, outside a run


class MallocMemoryTest(unittest.TestCase):
    """The cap without pymalloc: Emscripten (Pyodide) and WASI build --without-pymalloc, where the MEM and OBJ domains
    call malloc and never reach RAW or the arenas. PYTHONMALLOC=malloc is the same wiring on this build."""

    def run_memory_test(self, check, *args, **env):
        from test.support.script_helper import assert_python_ok
        code = ("import sys, unittest, _testinternalcapi, tracemalloc\n"
                f"{check}\n"
                "from test.test_restricted import MemoryTest\n"
                "r = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run("
                "unittest.defaultTestLoader.loadTestsFromTestCase(MemoryTest))\n"
                "sys.exit(not r.wasSuccessful())\n")
        assert_python_ok(*args, "-c", code, __isolated=False, **env)

    def test_memory_test_passes_under_malloc(self):
        self.run_memory_test("assert _testinternalcapi.pymem_getallocatorsname() == 'malloc', "
                             "_testinternalcapi.pymem_getallocatorsname()", PYTHONMALLOC="malloc")

    def test_memory_test_passes_under_tracemalloc(self):
        # tracemalloc hooks every domain at startup, so the allocator's name reads NULL; the startup choice decides
        # whether MEM and OBJ are wrapped. Before, NULL meant "wrap", and pymalloc counted each block twice.
        self.run_memory_test("assert tracemalloc.is_tracing()", "-X", "tracemalloc")

    def test_memory_test_passes_under_malloc_and_tracemalloc(self):
        self.run_memory_test("assert tracemalloc.is_tracing()", "-X", "tracemalloc", PYTHONMALLOC="malloc")

class Frame:
    """A stand-in for a pandas object: Python dunders that read internals the gate would refuse, and a __getitem__
    that calls a callable key, as pandas' does (df[f] is f(df))."""

    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __add__(self, other):
        return Frame(self._n + (other._n if isinstance(other, Frame) else other))

    def __getitem__(self, key):
        return key(self) if callable(key) else key


class PlainFrame(Frame):
    """The same, with its kind NOT listing "__^"."""


class LiftedDunderTest(unittest.TestCase):
    """A kind that lists "__^" has its Python dunders run lifted when a type slot reaches them (len, +, iteration),
    after the gate's own barrier; and what crosses into a lifted extent (a "[]^" item key or value, a lifted dunder's
    operand) obeys the callback rule: a callable must be the program's own function."""

    def setUp(self):
        for f in (len,):
            _restricted.allow_call(f, False)
        _restricted.allow_kind(Frame, ["[]", "[]^", "__^"])
        _restricted.allow_kind(PlainFrame, ["[]", "[]^"])
        _restricted.allow_kind(str, ["format"])

    def tearDown(self):
        _restricted.reset()

    def run_cell(self, src, **names):
        return restricted.run_cell(src, dict(names))

    def test_a_listed_kinds_dunders_run_lifted(self):
        self.assertEqual(self.run_cell("return len(f)", f=Frame(3)), 3)
        self.assertEqual(self.run_cell("return len(f + 2)", f=Frame(3)), 5)

    def test_an_unlisted_kinds_dunders_run_gated(self):
        with self.assertRaisesRegex(_restricted.NotInDialect, "_n"):
            self.run_cell("return len(f)", f=PlainFrame(3))

    def test_a_callable_key_must_be_the_programs_own(self):
        self.assertEqual(self.run_cell("return f[lambda x: 7]", f=Frame(1)), 7)
        for key in ('"{0.__class__}".format', "len"):
            with self.assertRaisesRegex(_restricted.NotInDialect, "not the program's own", msg=key):
                self.run_cell(f"return f[{key}]", f=Frame(1))

    def test_a_callable_inside_a_key(self):
        with self.assertRaisesRegex(_restricted.NotInDialect, "not the program's own"):
            self.run_cell('return f[("a", ["{0.__class__}".format])]', f=Frame(1))

    def test_a_class_as_a_key_is_data(self):
        self.assertTrue(self.run_cell("return len(f[str]) > 0", f=Frame(4)))

    def test_a_lifted_dunders_operand(self):
        with self.assertRaisesRegex(_restricted.NotInDialect, "not the program's own"):
            self.run_cell('return f + "{0.__class__}".format', f=Frame(1))


class UndoTest(unittest.TestCase):
    """Slice 6: a REFUSED run (the gate, the budget, the depth, the memory cap, or a refusal library code caught)
    leaves the namespace and every value it reaches as before the cell: bindings, and the contents of every value
    the cell wrote that existed before it. An ordinary error keeps its writes, as python_exec does today."""

    def setUp(self):
        for f in (len, list, range, sorted, dict, set, bytearray):
            _restricted.allow_call(f, False, restricted._range_cost if f is range else None)
        _restricted.allow_kind(list, ["append", "extend", "sort", "pop", "[]", "[]="])
        _restricted.allow_kind(dict, ["update", "pop", "[]", "[]="])
        _restricted.allow_kind(set, ["add"])
        _restricted.allow_kind(bytearray, ["append", "[]", "[]="])
        for t, names in restricted.MUTATORS.items():
            _restricted.allow_mutators(t, names)
        _restricted.allow_call(swallow, True)
        _restricted.allow_call(each, True)

    def tearDown(self):
        _restricted.reset()

    def cell(self, ns, src, **kw):
        ns.setdefault("swallow", swallow)
        ns.setdefault("each", each)
        return restricted.run_cell(src, ns, **kw)

    def world(self):
        ns = {}
        self.cell(ns, "xs = [3, 1, 2]\nd = {'a': 1}\ns = {1}\nb = bytearray(2)\nnested = [[1], {'k': [2]}]\nn = 1")
        return ns

    def snapshot(self, ns):
        return repr(sorted((k, v) for k, v in ns.items() if not callable(v)))

    REFUSALS = {
        "the gate": "().__class__",
        "the budget": "for i in range(1000):\n    pass",
        "the depth": "def f(n):\n    return f(n + 1)\nf(0)",
        "the memory cap": '"a" * 10 ** 11',
        "a caught refusal": "swallow(lambda: ().__class__)",
    }

    WRITES = ("xs.append(9)\nxs[0] = 7\nxs.sort()\nxs += [5]\n"
              "d['b'] = 2\nd.update({'a': 0})\ndel d['a']\n"
              "s.add(5)\ns |= {6}\nb[0] = 1\nb.append(3)\n"
              "nested[0].append(8)\nnested[1]['k'].append(9)\nnested[1]['z'] = 1\n"
              "n = 99\nnew_name = 1\ndel n\n")

    def test_every_write_before_every_kind_of_refusal_is_undone(self):
        for why, refusal in self.REFUSALS.items():
            ns = self.world()
            before = self.snapshot(ns)
            with self.assertRaises(_restricted.NotInDialect, msg=why):
                self.cell(ns, self.WRITES + refusal, budget=200 if why == "the budget" else None)
            self.assertEqual(self.snapshot(ns), before, why)

    def test_writes_inside_a_callback_from_a_lifted_leaf_are_undone(self):
        ns = self.world()
        before = self.snapshot(ns)
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "each(lambda i: xs.append(i), 5)\n().__class__")
        self.assertEqual(self.snapshot(ns), before)

    def test_the_same_value_written_many_times_is_logged_once(self):
        ns = self.world()
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "for i in range(10000):\n    xs.append(i)\n().__class__")
        self.assertEqual(ns["xs"], [3, 1, 2])

    def test_a_value_aliased_by_two_names(self):
        ns = self.world()
        self.cell(ns, "ys = xs")
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "ys.append(1)\nxs.append(2)\n().__class__")
        self.assertIs(ns["xs"], ns["ys"])
        self.assertEqual(ns["xs"], [3, 1, 2])

    def test_identity_is_kept(self):
        ns = self.world()
        xs = ns["xs"]
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "xs.append(1)\nxs = [0]\n().__class__")
        self.assertIs(ns["xs"], xs, "the binding is restored to the same object, not a copy")

    def test_a_successful_run_keeps_its_writes(self):
        ns = self.world()
        self.cell(ns, "xs.append(9)\nd['b'] = 2")
        self.assertEqual((ns["xs"], ns["d"]), ([3, 1, 2, 9], {"a": 1, "b": 2}))

    def test_an_ordinary_error_keeps_its_writes(self):
        ns = self.world()
        with self.assertRaises(ZeroDivisionError):
            self.cell(ns, "xs.append(9)\n1 // 0")
        self.assertEqual(ns["xs"], [3, 1, 2, 9])

    def test_the_log_does_not_leak_into_the_next_run(self):
        ns = self.world()
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "xs.append(1)\n().__class__")
        self.cell(ns, "xs.append(2)")
        self.assertEqual(ns["xs"], [3, 1, 2, 2])
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "xs.append(3)\n().__class__")
        self.assertEqual(ns["xs"], [3, 1, 2, 2], "a refusal undoes back to the last successful cell, no further")

    def test_values_this_cell_made_are_simply_dropped(self):
        ns = self.world()
        with self.assertRaises(_restricted.NotInDialect):
            self.cell(ns, "zs = []\nzs.append(1)\n().__class__")
        self.assertNotIn("zs", ns)


class PrintTest(unittest.TestCase):
    """print() in a cell: each argument converted by the cell's own gated str(), the lifted writer handed only
    strings, and nothing but the host's stream reachable. The corpus prints in a fifth of its cells."""

    def setUp(self):
        for f in (str, len, range, list):
            _restricted.allow_call(f, False)
        _restricted.allow_kind(list, ["[]"])
        _restricted.allow_kind(str, ["upper"])
        restricted.allow_print()

    def tearDown(self):
        _restricted.reset()

    def printed(self, src, **names):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            value = restricted.run_cell(src, dict(names))
        return out.getvalue(), value

    def test_print_writes_what_str_makes(self):
        self.assertEqual(self.printed("print(1, 'a', 2.5, None, [1, 'b'])")[0], "1 a 2.5 None [1, 'b']\n")
        self.assertEqual(self.printed("print('a', 'b', sep='-', end='!')")[0], "a-b!")
        self.assertEqual(self.printed("print()")[0], "\n")
        self.assertEqual(self.printed("xs = [1, 2]\nprint(*xs, sep=',')")[0], "1,2\n")
        self.assertEqual(self.printed("print(f'{len([1, 2])} rows', flush=True)")[0], "2 rows\n")

    def test_print_in_a_function_and_a_loop(self):
        out, value = self.printed("def show(x):\n    print(x.upper())\nfor i in range(3):\n    show('r' + str(i))\nreturn 7")
        self.assertEqual((out, value), ("R0\nR1\nR2\n", 7))

    def test_print_admits_only_what_str_admits(self):
        # a Python __str__ runs under the gate (its kind lists no "__^"): what str() refuses there, print refuses too
        class Secretive:
            def __str__(self):
                return str(self._hidden)
        import contextlib, io
        out = io.StringIO()
        for src in ("return str(s)", "print(s)", "print('first', s)"):
            with self.subTest(src=src), contextlib.redirect_stdout(out):
                with self.assertRaisesRegex(restricted.NotInDialect, "_hidden"):
                    restricted.run_cell(src, {"s": Secretive()})
        self.assertEqual(out.getvalue(), "")

    def test_no_stream_but_the_hosts(self):
        for src in ("print('x', file=xs)", "print(**kw)", "print('x', file=None)", "print('x', foo=1)"):
            with self.subTest(src=src), self.assertRaisesRegex(restricted.NotInDialect, "print"):
                self.printed(src, xs=[], kw={"file": []})

    def test_print_is_only_called(self):
        for src in ("p = print\np('x')", "list(map(print, ['x']))", "return print", "print = len\nreturn 1"):
            with self.subTest(src=src), self.assertRaisesRegex(restricted.NotInDialect, "print"):
                self.printed(src)

    def test_the_writer_takes_only_strings(self):
        # a cell that rebinds str gets its values to the writer unconverted: the writer's check refuses them
        _restricted.allow_call(int, False)
        with self.assertRaisesRegex(restricted.NotInDialect, "str"):
            self.printed("str = int\nprint(5)")
        with self.assertRaisesRegex(restricted.NotInDialect, "sep"):
            self.printed("print('a', sep=1)")
        with self.assertRaisesRegex(restricted.NotInDialect, "flush"):
            self.printed("print('a', flush=1)")

    def test_the_writer_is_not_the_cells_name(self):
        with self.assertRaisesRegex(restricted.NotInDialect, "__cell_print__"):
            self.printed("__cell_print__('x')")
        ns = {}
        restricted.run_cell("return 1", ns)
        self.assertNotIn("__cell_print__", ns)

    def test_output_before_a_refusal_has_already_left(self):
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(restricted.NotInDialect):
            restricted.run_cell("print('before')\nreturn box.secret", {"box": Box()})
        self.assertEqual(out.getvalue(), "before\n")

    def test_printing_in_a_loop_is_metered(self):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(restricted.NotInDialect, "budget"):
            restricted.run_cell("for i in range(1000000):\n    print(i)", {}, budget=1000)

try:
    import warnings
    with warnings.catch_warnings():     # numpy adds filters at import; keep the test environment as it was
        import numpy
        import pandas
except ImportError:
    numpy = pandas = None


@unittest.skipIf(pandas is None, "needs numpy and pandas (stock wheels) installed for this build")
class StockWheelsTest(unittest.TestCase):
    """The ABI rule: stock numpy and pandas wheels, built for stock CPython, import and run under the patched binary,
    and run lifted from a restricted cell: methods and dunders, ownership of a copy, and the barriers."""

    def setUp(self):
        pd = pandas
        for f in (len, int, list, sum):
            _restricted.allow_call(f, False)
        _restricted.allow_module("pandas", ["DataFrame"])
        _restricted.allow_call(pd.DataFrame, True)
        _restricted.allow_fresh(pd.DataFrame)
        for cls in (pd.DataFrame, pd.Series):
            _restricted.allow_kind(cls, ["sum", "copy", "[]", "[]=", "[]^", "__^"])
            _restricted.allow_call(cls.sum, True)
            _restricted.allow_call(cls.copy, True)
            _restricted.allow_fresh(cls.copy)
        _restricted.allow_kind(numpy.int64, [])
        _restricted.allow_kind(str, ["format"])

    def tearDown(self):
        _restricted.reset()

    def df(self):
        return pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def run_cell(self, src, **names):
        return restricted.run_cell(src, dict(names))

    def test_print_a_frame(self):
        import contextlib, io
        restricted.allow_print()
        _restricted.allow_call(str, False)
        out = io.StringIO()
        df = self.df()
        with contextlib.redirect_stdout(out):
            self.run_cell("print(df)\nprint(df['a'].sum(), len(df))", df=df)
        self.assertEqual(out.getvalue(), str(df) + "\n6 3\n")

    def test_the_wheels_are_stock(self):
        self.assertFalse(any(_restricted.build().values()))
        self.assertTrue(numpy.__file__ and pandas.__file__)

    def test_methods_and_dunders(self):
        df = self.df()
        self.assertEqual(self.run_cell("return int(df['a'].sum())", df=df), 6)
        self.assertEqual(self.run_cell("return len(df)", df=df), 3)
        self.assertEqual(self.run_cell("return int((df['a'] * 2).sum())", df=df), 12)
        self.assertEqual(self.run_cell("return int((df['a'] > 1).sum())", df=df), 2)
        self.assertEqual(self.run_cell("return [c for c in df]", df=df), ["a", "b"])

    def test_an_injected_frame_is_not_the_cells(self):
        df = self.df()
        for src in ("df['c'] = 1", "df += 1", "df['a'] += 1"):
            with self.assertRaises(_restricted.NotInDialect, msg=src):
                self.run_cell(src, df=df)
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(int(df["a"].sum()), 6)

    def test_a_copy_is_the_cells(self):
        df = self.df()
        self.assertEqual(self.run_cell("d = df.copy()\nd['c'] = 1\nd += 1\nreturn int(d['c'].sum())", df=df), 6)
        self.assertEqual(list(df.columns), ["a", "b"])

    def test_an_owned_frame_from_an_earlier_cell_is_restored(self):
        _restricted.allow_snapshot(pandas.DataFrame, restricted._snapshot_frame, restricted._restore_frame)
        _restricted.allow_snapshot(pandas.Series, restricted._snapshot_frame, restricted._restore_frame)
        ns = {"df": self.df()}
        restricted.run_cell("d = df.copy()", ns)
        with self.assertRaises(_restricted.NotInDialect):
            restricted.run_cell("d['c'] = 1\nd['a'] = 0\nd += 1\n().__class__", ns)
        self.assertEqual(list(ns["d"].columns), ["a", "b"])
        self.assertEqual(ns["d"]["a"].tolist(), [1, 2, 3])

    def test_a_callable_key(self):
        df = self.df()
        self.assertEqual(self.run_cell("return int(df[lambda d: d['a'] > 1]['a'].sum())", df=df), 5)
        with self.assertRaisesRegex(_restricted.NotInDialect, "not the program's own"):
            self.run_cell('return df["{0.__class__}".format]', df=df)


if __name__ == "__main__":
    unittest.main()
