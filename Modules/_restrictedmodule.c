/* _restricted: window.ml's restricted mode, the entry points.

   A cell the read-only dialect admitted is compiled with CO_RESTRICTED on every code object (Lib/restricted.py) and
   run through run(), which marks the thread restricted for the C extent of the call. The gate in the abstract object
   layer (a later slice) consults _PyRestricted_Active().

   This module is host-only: restricted code must never reach it (the import gate refuses it), since call_lifted()
   is a way out of the restriction by design. */

#ifndef Py_BUILD_CORE_BUILTIN
#  define Py_BUILD_CORE_MODULE 1
#endif

#include "Python.h"
#include "pycore_code.h"        // ENABLE_SPECIALIZATION
#include "pycore_restricted.h"  // _PyRestricted_Active()
#include "pycore_ceval.h"        // _PY_EVAL_RESTRICTED_BIT
#include "pycore_pystate.h"      // _PyThreadState_GET()
#include "pycore_tstate.h"      // _PyThreadStateImpl

/* The step budget a run gets when the caller names none: about a second of interpreter work (measured: see
   DEFAULT_BUDGET in Lib/restricted.py), the same bound the JS dialect's STEP_BUDGET sets. */
#define RESTRICTED_DEFAULT_BUDGET 15000000LL
/* How deep a program's own frames may sit above where the run began, in Python frames (the JS dialect's
   MAX_CALL_DEPTH). */
#define RESTRICTED_DEFAULT_DEPTH 256
/* Bytes a run may add to the heap, net (the RAW allocator's blocks and pymalloc's arenas: Python/restricted.c). */
#define RESTRICTED_DEFAULT_MEMORY (256LL << 20)
/* Frames left below the depth for CPython's own limit, so the dialect's refusal comes before a RecursionError. */
#define RESTRICTED_DEPTH_MARGIN 50

static long long last_used;
static long long last_peak;     /* the most heap the last outermost run held at once */
static PyObject *not_in_dialect;  /* borrowed: the module keeps it alive; set at init */     /* steps the last finished run used (one interpreter per process, as the gate) */

/* run(code, globals, budget=None, depth=None, memory=None, /): evaluate a CO_RESTRICTED code object in `globals`, with the thread
   restricted, metered and depth-bounded for the call. The previous state is restored on every exit, an exception
   included, because this is a C call. A run during which anything was refused ends in NotInDialect, however it ended:
   library code that caught the refusal cannot turn it into a value. */
static PyObject *
restricted_run(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs < 2 || nargs > 5) {
        PyErr_SetString(PyExc_TypeError,
                        "run() takes 2 to 5 arguments (code, globals, budget=None, depth=None, memory=None)");
        return NULL;
    }
    PyObject *code = args[0], *globals = args[1];
    if (!PyCode_Check(code)) {
        PyErr_SetString(PyExc_TypeError, "run() needs a code object");
        return NULL;
    }
    if (!(((PyCodeObject *)code)->co_flags & CO_RESTRICTED)) {
        PyErr_SetString(PyExc_ValueError, "run() needs code compiled by restricted.compile() (CO_RESTRICTED)");
        return NULL;
    }
    if (!PyDict_Check(globals)) {
        PyErr_SetString(PyExc_TypeError, "run() needs a dict of globals");
        return NULL;
    }
    long long budget = RESTRICTED_DEFAULT_BUDGET;
    long depth = RESTRICTED_DEFAULT_DEPTH;
    if (nargs > 2 && args[2] != Py_None) {
        budget = PyLong_AsLongLong(args[2]);
        if (budget == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (budget < 0) {
            PyErr_SetString(PyExc_ValueError, "run() needs a budget >= 0");
            return NULL;
        }
    }
    if (nargs > 3 && args[3] != Py_None) {
        depth = PyLong_AsLong(args[3]);
        if (depth == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (depth < 1) {
            PyErr_SetString(PyExc_ValueError, "run() needs a depth >= 1");
            return NULL;
        }
    }
    long long memory = RESTRICTED_DEFAULT_MEMORY;
    if (nargs > 4 && args[4] != Py_None) {
        memory = PyLong_AsLongLong(args[4]);
        if (memory == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (memory < 0) {
            PyErr_SetString(PyExc_ValueError, "run() needs a memory cap >= 0");
            return NULL;
        }
    }
    if (_PyRestricted_OwnedReady() < 0) {
        return NULL;
    }
    PyThreadState *tstate = _PyThreadState_GET();
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    int room = tstate->py_recursion_remaining - RESTRICTED_DEPTH_MARGIN;
    if (depth > room) {
        depth = room > 1 ? room : 1;
    }

    _PyRestrictedSaved saved = {ts->restricted, ts->restricted_lifted, ts->restricted_barrier};
    long long saved_steps = ts->restricted_steps, saved_budget = ts->restricted_budget;
    int saved_base = ts->restricted_depth_base, saved_limit = ts->restricted_depth_limit;
    int saved_tripped = ts->restricted_tripped;
    int was_metered = (_Py_atomic_load_uintptr_relaxed(&tstate->eval_breaker) & _PY_EVAL_RESTRICTED_BIT) != 0;

    ts->restricted = 1;
    ts->restricted_lifted = 0;
    ts->restricted_barrier = NULL;
    ts->restricted_steps = budget;
    ts->restricted_budget = budget;
    ts->restricted_depth_base = tstate->py_recursion_remaining;
    ts->restricted_depth_limit = (int)depth;
    ts->restricted_tripped = 0;
    _Py_set_eval_breaker_bit(tstate, _PY_EVAL_RESTRICTED_BIT);
    int counting = _PyRestricted_MemoryBegin(memory);
    int logging = counting < 0 ? -1 : _PyRestricted_UndoBegin();
    /* The namespace itself: a cell's top-level names are globals (STORE_GLOBAL writes the dict directly, past the
       gate), so a refused run puts the bindings back as they were, new names removed. Only the outermost run. */
    PyObject *names_before = logging > 0 ? PyDict_Copy(globals) : NULL;
    if (logging > 0 && names_before == NULL) {
        logging = -1;
    }
    PyObject *result = logging < 0 ? NULL : PyEval_EvalCode(code, globals, globals);

    if (!was_metered) {
        _Py_unset_eval_breaker_bit(tstate, _PY_EVAL_RESTRICTED_BIT);
    }
    int memory_hit = 0;
    if (counting > 0) {
        memory_hit = _PyRestricted_MemoryEnd(&last_peak);
    }
    if (memory_hit) {
        /* The cap is the refusal, whatever the failed allocation turned into on the way out. */
        Py_CLEAR(result);
        PyObject *exc = PyErr_GetRaisedException();
        PyErr_Format(not_in_dialect, "not in dialect: the cell's memory cap (%lld bytes) ran out", memory);
        if (exc != NULL) {
            PyObject *now = PyErr_GetRaisedException();
            PyException_SetContext(now, exc);
            PyErr_SetRaisedException(now);
        }
    }
    last_used = ts->restricted_steps < 0 ? budget + 1 : budget - ts->restricted_steps;
    if (ts->restricted_tripped && !memory_hit) {
        PyObject *exc = PyErr_GetRaisedException();
        if (exc == NULL || !PyErr_GivenExceptionMatches(exc, not_in_dialect)) {
            /* Library code caught the refusal and went on (to a value, or to another error). */
            Py_CLEAR(result);
            PyErr_SetString(not_in_dialect,
                            "not in dialect: a refusal during the run was caught by library code");
            if (exc != NULL) {
                PyObject *now = PyErr_GetRaisedException();
                PyException_SetContext(now, exc);   /* steals exc */
                PyErr_SetRaisedException(now);
            }
        }
        else {
            PyErr_SetRaisedException(exc);
        }
    }
    ts->restricted = saved.restricted;
    ts->restricted_lifted = saved.lifted;
    ts->restricted_barrier = saved.barrier;
    if (logging >= 0) {
        /* A REFUSED run (NotInDialect, however it arose: the gate, the budget, the depth, the memory cap) replays the
           undo log, so the namespace and every value it reaches are as before the cell. An ordinary error (a
           ZeroDivisionError in the cell's own logic) keeps its writes, as python_exec does today. */
        int refused = result == NULL && PyErr_ExceptionMatches(not_in_dialect);
        int replay = _PyRestricted_UndoEnd(refused);
        if (refused && names_before != NULL) {
            PyObject *exc = PyErr_GetRaisedException();
            PyDict_Clear(globals);
            if (PyDict_Update(globals, names_before) < 0) {
                PyErr_Clear();
                replay = -1;
            }
            PyErr_SetRaisedException(exc);
        }
        if (replay < 0 && refused) {
            PyObject *exc = PyErr_GetRaisedException();
            PyErr_SetString(PyExc_SystemError, "restricted: a refused run's undo log did not replay cleanly");
            PyObject *now = PyErr_GetRaisedException();
            PyException_SetContext(now, exc);
            PyErr_SetRaisedException(now);
        }
    }
    Py_XDECREF(names_before);
    ts->restricted_steps = saved_steps;
    ts->restricted_budget = saved_budget;
    ts->restricted_depth_base = saved_base;
    ts->restricted_depth_limit = saved_limit;
    ts->restricted_tripped = saved_tripped;
    return result;
}

/* allow_mutators(type, names): these methods write their receiver; restricted code reads them only off a value it
   owns. */
static PyObject *
restricted_allow_mutators(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2 || !PyType_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "allow_mutators(type, names)");
        return NULL;
    }
    if (_PyRestricted_AllowMutators(args[0], args[1]) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* allow_fresh(callable): a lifted leaf whose result is a new value, owned by the cell (the manifest's `fresh`). */
static PyObject *
restricted_allow_fresh(PyObject *module, PyObject *callable)
{
    if (_PyRestricted_AllowFresh(callable) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* allow_snapshot(type, snapshot, restore): how the undo log copies and restores an owned value of a kind CPython
   cannot copy itself (a pandas frame, a numpy array). snapshot(obj) -> state; restore(obj, state) in place. */
static PyObject *
restricted_allow_snapshot(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3 || !PyType_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "allow_snapshot(type, snapshot, restore)");
        return NULL;
    }
    if (_PyRestricted_AllowSnapshot(args[0], args[1], args[2]) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* write_barrier(obj): log obj's contents before a lifted leaf writes it (an entry checker calls this). */
static PyObject *
restricted_write_barrier(PyObject *module, PyObject *obj)
{
    if (_PyRestricted_WriteBarrier(obj) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* owned(obj): whether obj is the cell's own (an entry checker asks before a leaf writes its receiver). */
static PyObject *
restricted_owned(PyObject *module, PyObject *obj)
{
    return PyBool_FromLong(_PyRestricted_IsOwned(obj));
}

/* _table_sizes(): (len(owned set), len(lifted generators)), for tests. */
static PyObject *
restricted_table_sizes(PyObject *module, PyObject *unused)
{
    Py_ssize_t a, b;
    _PyRestricted_TableSizes(&a, &b);
    return Py_BuildValue("(nn)", a, b);
}

/* memory(): the most heap the last run held at once, in bytes (RAW blocks and arenas). */
static PyObject *
restricted_memory(PyObject *module, PyObject *unused)
{
    return PyLong_FromLongLong(last_peak);
}

/* steps(): the steps the last finished run used (budget + 1 when it ran out). */
static PyObject *
restricted_steps(PyObject *module, PyObject *unused)
{
    return PyLong_FromLongLong(last_used);
}

/* call_lifted(callable, *args): call with the restriction lifted for the call's C extent, the way an approved C leaf
   will run (a later slice decides approval from the manifest). A CO_RESTRICTED frame the call reaches is restricted
   again. Host-only: exposed for tests. */
static PyObject *
restricted_call_lifted(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs < 1) {
        PyErr_SetString(PyExc_TypeError, "call_lifted() needs a callable");
        return NULL;
    }
    PyThreadState *tstate = _PyThreadState_GET();
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    int saved = ts->restricted, saved_lifted = ts->restricted_lifted;
    struct _PyInterpreterFrame *saved_barrier = ts->restricted_barrier;
    ts->restricted = 0;
    ts->restricted_lifted = 1;
    ts->restricted_barrier = tstate->current_frame;
    PyObject *result = PyObject_Vectorcall(args[0], args + 1, nargs - 1, NULL);
    ts->restricted = saved;
    ts->restricted_lifted = saved_lifted;
    ts->restricted_barrier = saved_barrier;
    return result;
}

/* active(): whether the gate is active for this thread now. */
static PyObject *
restricted_active(PyObject *module, PyObject *Py_UNUSED(ignored))
{
    return PyBool_FromLong(_PyRestricted_Active(_PyThreadState_GET()));
}

/* build(): what this interpreter was built with, read at run time: a runner refuses to run the vectors unless
   specialization, tier 2 and the JIT are all off (repl-specs 2026-10-15). */
static PyObject *
restricted_build(PyObject *module, PyObject *Py_UNUSED(ignored))
{
#ifdef _Py_TIER2
    int tier2 = 1;
#else
    int tier2 = 0;
#endif
#ifdef _Py_JIT
    int jit = 1;
#else
    int jit = 0;
#endif
    return Py_BuildValue("{sOsOsOsO}",
                         "specialization", ENABLE_SPECIALIZATION ? Py_True : Py_False,
                         "specialization_ft", ENABLE_SPECIALIZATION_FT ? Py_True : Py_False,
                         "tier2", tier2 ? Py_True : Py_False,
                         "jit", jit ? Py_True : Py_False);
}

/* allow_kind(type, names): a restricted program may read these attributes on an instance of exactly `type`
   ("[]" item reads, "[]=" item writes). Replaces any earlier entry for the type. */
static PyObject *
restricted_allow_kind(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2 || !PyType_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "allow_kind(type, names)");
        return NULL;
    }
    if (_PyRestricted_AllowKind(args[0], args[1]) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* allow_module(name, names): importing `name` is allowed, and reading these attributes of it. */
static PyObject *
restricted_allow_module(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2 || !PyUnicode_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "allow_module(name, names)");
        return NULL;
    }
    if (_PyRestricted_AllowModule(args[0], args[1]) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* allow_call(callable, lift=False): a restricted program may call it. Unlifted by default, so its own lookups are
   gated; lift=True only for a manifest leaf whose internals legitimately touch what the gate refuses, and NEVER for
   one that looks up names it is given (format, getattr, attrgetter). */
static PyObject *
restricted_allow_call(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs < 1 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "allow_call(callable, lift=False, checker=None)");
        return NULL;
    }
    int lift = nargs >= 2 ? PyObject_IsTrue(args[1]) : 0;
    PyObject *checker = nargs == 3 && args[2] != Py_None ? args[2] : NULL;
    /* Lifted, the checker is the leaf's ENTRY check; unlifted, it is a COST check (a C loop the budget cannot see,
       bounded by its arguments). Both run before the call, lifted, and refuse by raising. */
    if (lift < 0 || _PyRestricted_AllowCall(args[0], lift, checker) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* allow_dynamic(type, predicate): for a name outside the type's table, predicate(obj, name), run lifted, may admit it;
   the read is then lifted. Never consulted for a dunder or a listed name (a manifest's `dynamic: columns`). */
static PyObject *
restricted_allow_dynamic(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2 || !PyType_Check(args[0]) || !PyCallable_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "allow_dynamic(type, predicate)");
        return NULL;
    }
    if (_PyRestricted_AllowDynamic(args[0], args[1]) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

/* reset(): forget every allow table (tests). */
static PyObject *
restricted_reset(PyObject *module, PyObject *Py_UNUSED(ignored))
{
    _PyRestricted_Reset();
    Py_RETURN_NONE;
}

static PyMethodDef restricted_methods[] = {
    {"allow_kind", _PyCFunction_CAST(restricted_allow_kind), METH_FASTCALL,
     "allow_kind(type, names, /)\n--\n\nAllow reading these attributes on instances of exactly type."},
    {"allow_module", _PyCFunction_CAST(restricted_allow_module), METH_FASTCALL,
     "allow_module(name, names, /)\n--\n\nAllow importing name and reading these of its attributes."},
    {"allow_call", _PyCFunction_CAST(restricted_allow_call), METH_FASTCALL,
     "allow_call(callable, lift=False, checker=None, /)\n--\n\nAllow calling it from restricted code; lift=True only for a manifest leaf.\nchecker(func, self, args, kwargs) runs first: the leaf's entry check, or an\nunlifted callable's cost check."},
    {"allow_dynamic", _PyCFunction_CAST(restricted_allow_dynamic), METH_FASTCALL,
     "allow_dynamic(type, predicate, /)\n--\n\nAdmit names outside type's table when predicate(obj, name) is true."},
    {"reset", restricted_reset, METH_NOARGS,
     "reset()\n--\n\nForget every allow table."},
    {"run", _PyCFunction_CAST(restricted_run), METH_FASTCALL,
     "run(code, globals, budget=None, depth=None, memory=None, /)\n--\n\nEvaluate a CO_RESTRICTED code object with the thread restricted, metered by\na step budget, bounded in depth and capped in memory."},
    {"allow_mutators", _PyCFunction_CAST(restricted_allow_mutators), METH_FASTCALL,
     "allow_mutators(type, names, /)\n--\n\nThese methods write their receiver: read only off an owned value."},
    {"allow_fresh", restricted_allow_fresh, METH_O,
     "allow_fresh(callable, /)\n--\n\nA lifted leaf whose result is a new value, owned by the cell."},
    {"allow_snapshot", _PyCFunction_CAST(restricted_allow_snapshot), METH_FASTCALL,
     "allow_snapshot(type, snapshot, restore, /)\n--\n\nHow the undo log copies and restores an owned value of this kind."},
    {"write_barrier", restricted_write_barrier, METH_O,
     "write_barrier(obj, /)\n--\n\nLog obj's contents before a lifted leaf writes it."},
    {"owned", restricted_owned, METH_O,
     "owned(obj, /)\n--\n\nWhether obj is the cell's own."},
    {"_table_sizes", restricted_table_sizes, METH_NOARGS,
     "_table_sizes()\n--\n\nThe sizes of the owned set and of the lifted generators' table (tests)."},
    {"memory", restricted_memory, METH_NOARGS,
     "memory()\n--\n\nThe most heap the last run held at once, in bytes."},
    {"steps", restricted_steps, METH_NOARGS,
     "steps()\n--\n\nThe steps the last finished run used (budget + 1 when it ran out)."},
    {"call_lifted", _PyCFunction_CAST(restricted_call_lifted), METH_FASTCALL,
     "call_lifted(callable, *args)\n--\n\nCall with the restriction lifted, as an approved C leaf runs (tests only)."},
    {"active", restricted_active, METH_NOARGS,
     "active()\n--\n\nWhether the restricted-mode gate is active for this thread."},
    {"build", restricted_build, METH_NOARGS,
     "build()\n--\n\nWhether this interpreter specializes, has tier 2 or a JIT (all must be False)."},
    {NULL, NULL, 0, NULL}
};

static int
restricted_exec(PyObject *module)
{
    if (PyModule_Add(module, "DEFAULT_BUDGET", PyLong_FromLongLong(RESTRICTED_DEFAULT_BUDGET)) < 0
        || PyModule_AddIntConstant(module, "DEFAULT_DEPTH", RESTRICTED_DEFAULT_DEPTH) < 0
        || PyModule_Add(module, "DEFAULT_MEMORY", PyLong_FromLongLong(RESTRICTED_DEFAULT_MEMORY)) < 0)
    {
        return -1;
    }
    if (PyModule_AddIntConstant(module, "CO_RESTRICTED", CO_RESTRICTED) < 0) {
        return -1;
    }
    /* NotInDialect: a refusal by the gate. A subclass of PermissionError, so code that already treats a refused
       operation as a permission problem keeps working; the host tells the two apart by type. */
    PyObject *exc = PyErr_NewExceptionWithDoc("_restricted.NotInDialect",
        "The restricted-mode gate refused an operation: the program is outside the read-only dialect.",
        PyExc_PermissionError, NULL);
    if (exc == NULL) {
        return -1;
    }
    _PyRestricted_SetException(exc);
    not_in_dialect = exc;
    return PyModule_Add(module, "NotInDialect", exc);
}

static PyModuleDef_Slot restricted_slots[] = {
    {Py_mod_exec, restricted_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, NULL},
};

static struct PyModuleDef restricted_module = {
    PyModuleDef_HEAD_INIT,
    "_restricted",
    "window.ml restricted mode: the entry points (host-only).",
    0,
    restricted_methods,
    restricted_slots,
};

PyMODINIT_FUNC
PyInit__restricted(void)
{
    return PyModuleDef_Init(&restricted_module);
}
