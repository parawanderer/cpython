"""Tests for window.ml's restricted mode, slice 1: the flag, the entry, and the dynamic extent.

The gate itself (refusing attributes, items, calls and imports) is a later slice; here ``_restricted.active()`` is
the observable, called from inside the code under test. The dialect vectors (window-ml's
tests/fixtures/dialect-vectors/) are the acceptance suite once the gate exists.
"""

import unittest

import _restricted
import restricted


def host_active():
    """Host Python: not restricted code itself."""
    return _restricted.active()


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

    def test_a_top_level_return_is_the_value(self):
        self.assertEqual(restricted.run_cell("return 6 * 7", {}), 42)


class ExtentTest(unittest.TestCase):
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
        restricted.run_cell("global probe\ndef probe():\n    return active()\nreturn probe()", ns)
        self.assertFalse(ns["probe"]())

    def test_runs_nest(self):
        ns = self.ns()
        ns["inner"] = lambda: restricted.run_cell("return active()", self.ns())
        self.assertEqual(restricted.run_cell("return [lift(inner), active()]", ns), [True, True])


if __name__ == "__main__":
    unittest.main()
