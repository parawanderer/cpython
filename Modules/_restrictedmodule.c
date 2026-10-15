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
#include "pycore_pystate.h"      // _PyThreadState_GET()
#include "pycore_tstate.h"      // _PyThreadStateImpl

/* run(code, globals, /): evaluate a CO_RESTRICTED code object in `globals`, with the thread restricted for the call.
   The previous state is restored on every exit, an exception included, because this is a C call. */
static PyObject *
restricted_run(PyObject *module, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "run() takes exactly 2 arguments (code, globals)");
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
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)_PyThreadState_GET();
    int saved = ts->restricted, saved_lifted = ts->restricted_lifted;
    struct _PyInterpreterFrame *saved_barrier = ts->restricted_barrier;
    ts->restricted = 1;
    ts->restricted_lifted = 0;
    ts->restricted_barrier = NULL;
    PyObject *result = PyEval_EvalCode(code, globals, globals);
    ts->restricted = saved;
    ts->restricted_lifted = saved_lifted;
    ts->restricted_barrier = saved_barrier;
    return result;
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

static PyMethodDef restricted_methods[] = {
    {"run", _PyCFunction_CAST(restricted_run), METH_FASTCALL,
     "run(code, globals, /)\n--\n\nEvaluate a CO_RESTRICTED code object with the thread restricted."},
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
    return PyModule_AddIntConstant(module, "CO_RESTRICTED", CO_RESTRICTED);
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
