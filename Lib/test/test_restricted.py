"""Tests for window.ml's restricted mode, slice 1: the flag, the entry, and the dynamic extent.

The gate itself (refusing attributes, items, calls and imports) is a later slice; here ``_restricted.active()`` is
the observable, called from inside the code under test. The dialect vectors (window-ml's
tests/fixtures/dialect-vectors/) are the acceptance suite once the gate exists.
"""

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

    def test_a_restricted_def_called_from_outside_any_entry_runs_unrestricted(self):
        # Ruled 2026-10-15: restriction is a property of the ENTRY; the bare caller already holds full capability.
        ns = self.ns()
        restricted.run_cell("def probe():\n    return active()\nreturn probe()", ns)
        self.assertFalse(ns["probe"]())

    def test_runs_nest(self):
        ns = self.ns()
        ns["inner"] = lambda: restricted.run_cell("return active()", self.ns())
        self.assertEqual(restricted.run_cell("return [lift(inner), active()]", ns), [True, True])


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


if __name__ == "__main__":
    unittest.main()
