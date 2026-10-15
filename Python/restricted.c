/* window.ml restricted mode: the gate.

   While _PyRestricted_Active() holds for the thread, the abstract object layer and the interpreter's generic call
   path consult the ALLOW tables here, which the host registers from the manifests through the _restricted module:

     kinds     type -> frozenset of attribute names a restricted program may read on an instance of exactly that type
               ("[]" for item reads, "[]=" for item writes)
     modules   module name -> frozenset of attribute names
     calls     callables a restricted program may call. By DEFAULT a C callable runs UNLIFTED: its own attribute, item
               and import lookups go through the gate, which is what covers the C paths (str.format walking
               "{0.__class__}", getattr, operator.attrgetter) by construction. Only a manifest leaf whose internals
               legitimately touch what the gate refuses (numpy, pandas) is registered LIFTED: the gate is off for its C
               extent, and a CO_RESTRICTED frame it calls back is gated again (pycore_restricted.h). Lifting a callable
               that does data-driven lookups (format, getattr, attrgetter) would open the C paths, so it must never be
               registered lifted.

   Everything not listed is refused with NotInDialect. A dunder is refused before any table is read. Python functions
   (the cell's own, and pure-Python library code: tier 3) are called without lifting, so they run under the gate.

   One registry per process for now (the Pyodide worker and the host each run one interpreter). */

#include "Python.h"
#include "pycore_pystate.h"       // _PyThreadState_GET()
#include "pycore_restricted.h"    // _PyRestricted_Active()
#include "pycore_tstate.h"        // _PyThreadStateImpl

static PyObject *kinds;           // dict: type -> frozenset[str]
static PyObject *modules;         // dict: str -> frozenset[str]
static PyObject *calls;           // set of callables, called lifted
static PyObject *calls_unlifted;  // set of callables called WITHOUT lifting (pure probes such as _restricted.active)
static PyObject *not_in_dialect;  // the exception type, registered by _restricted

static int
is_dunder(PyObject *name)
{
    if (!PyUnicode_Check(name)) {
        return 0;
    }
    Py_ssize_t n = PyUnicode_GET_LENGTH(name);
    return n >= 4
        && PyUnicode_READ_CHAR(name, 0) == '_' && PyUnicode_READ_CHAR(name, 1) == '_'
        && PyUnicode_READ_CHAR(name, n - 1) == '_' && PyUnicode_READ_CHAR(name, n - 2) == '_';
}

static int
refuse(const char *what, PyObject *obj, PyObject *name)
{
    PyObject *exc = not_in_dialect ? not_in_dialect : PyExc_PermissionError;
    if (name != NULL) {
        PyErr_Format(exc, "not in dialect: %s %R of %.100s", what, name, Py_TYPE(obj)->tp_name);
    }
    else {
        PyErr_Format(exc, "not in dialect: %s on %.100s", what, Py_TYPE(obj)->tp_name);
    }
    return -1;
}

/* Is `key` in the frozenset registered for `table[index]`? -1 on error. */
static int
listed(PyObject *table, PyObject *index, PyObject *key)
{
    if (table == NULL) {
        return 0;
    }
    PyObject *names;
    int r = PyDict_GetItemRef(table, index, &names);
    if (r <= 0) {
        return r;
    }
    r = PySet_Contains(names, key);
    Py_DECREF(names);
    return r;
}

static PyObject *item_read, *item_write, *call_marker, *item_lift;   // interned "[]", "[]=", "()" and "[]^"

static int
check_attr(PyObject *obj, PyObject *name, const char *what)
{
    if (is_dunder(name)) {
        return refuse(what, obj, name);
    }
    int r;
    if (PyModule_Check(obj)) {
        PyObject *modname = PyModule_GetNameObject(obj);
        if (modname == NULL) {
            PyErr_Clear();
            return refuse(what, obj, name);
        }
        r = listed(modules, modname, name);
        Py_DECREF(modname);
    }
    else {
        r = listed(kinds, (PyObject *)Py_TYPE(obj), name);
    }
    if (r < 0) {
        return -1;
    }
    return r ? 0 : refuse(what, obj, name);
}

/* Each returns 0 to proceed, -1 with NotInDialect set. Called only when _PyRestricted_Active(). */

int
_PyRestricted_CheckGetAttr(PyObject *obj, PyObject *name)
{
    return check_attr(obj, name, "attribute");
}

int
_PyRestricted_CheckSetAttr(PyObject *obj, PyObject *name)
{
    /* No restricted program sets an attribute in the first slice: it builds no classes, and a write to anything it
       did not create is a write to heap(unowned). Ownership (a later slice) may admit some. */
    return refuse("setting attribute", obj, name);
}

/* 0 to proceed unlifted, 1 to proceed LIFTED (the kind lists "[]^": its item protocol is a manifest leaf, e.g. a
   DataFrame's __getitem__ in pandas Python), -1 refused. */
int
_PyRestricted_CheckItem(PyObject *obj, int write)
{
    int r = listed(kinds, (PyObject *)Py_TYPE(obj), write ? item_write : item_read);
    if (r < 0) {
        return -1;
    }
    if (!r) {
        return refuse(write ? "item assignment" : "item access", obj, NULL);
    }
    r = listed(kinds, (PyObject *)Py_TYPE(obj), item_lift);
    return r < 0 ? -1 : r;
}

int
_PyRestricted_CheckImport(PyObject *name)
{
    if (modules != NULL) {
        int r = PyDict_Contains(modules, name);
        if (r != 0) {
            return r > 0 ? 0 : -1;
        }
    }
    PyErr_Format(not_in_dialect ? not_in_dialect : PyExc_PermissionError, "not in dialect: import %R", name);
    return -1;
}

/* Is a call to `callable` allowed, and must it be lifted? 1 = allowed and lifted (a listed C callable), 0 = allowed
   unlifted (a Python function: tier 3 or the program's own), -1 = refused (exception set). */
int
_PyRestricted_CheckCall(PyObject *callable)
{
    /* A manifest leaf registered lifted: the callable itself, or (a bound method is a fresh object per access) the
       function under it. Checked first, so a lifted pure-Python leaf (pandas) is lifted rather than run as tier 3. */
    if (calls != NULL) {
        PyObject *key = PyMethod_Check(callable) ? PyMethod_GET_FUNCTION(callable) : callable;
        int r = PySet_Contains(calls, key);
        if (r != 0) {
            return r > 0 ? 1 : -1;
        }
    }
    if (PyFunction_Check(callable) || PyMethod_Check(callable)) {
        return 0;
    }
    if (calls_unlifted != NULL) {
        int r = PySet_Contains(calls_unlifted, callable);
        if (r != 0) {
            return r > 0 ? 0 : -1;
        }
    }
    /* An instance of a kind that lists "()" (operator.attrgetter(...), a numpy ufunc): called unlifted. */
    if (kinds != NULL && call_marker != NULL) {
        int r = listed(kinds, (PyObject *)Py_TYPE(callable), call_marker);
        if (r != 0) {
            return r > 0 ? 0 : -1;
        }
    }
    /* A builtin method bound to a receiver, or an unbound method descriptor (what a method call's LOAD_ATTR hands
       CALL): reachable only through an attribute the gate already allowed. Runs UNLIFTED, so a lookup it makes from
       its arguments (str.format's field names) is gated. */
    if (Py_IS_TYPE(callable, &PyMethodDescr_Type)) {
        return 0;
    }
    if (PyCFunction_Check(callable) && PyCFunction_GET_SELF(callable) != NULL) {
        PyObject *self = PyCFunction_GET_SELF(callable);
        if (!PyModule_Check(self)) {
            return 0;
        }
        /* A module's C function (math.sqrt, but also builtins.eval, reached by name without an attribute read): only
           when that module's table lists it. */
        const char *fname = ((PyCFunctionObject *)callable)->m_ml->ml_name;
        PyObject *modname = PyModule_GetNameObject(self);
        if (modname == NULL) {
            PyErr_Clear();
            return refuse("call", callable, NULL);
        }
        PyObject *name = PyUnicode_FromString(fname);
        int r = name == NULL ? -1 : listed(modules, modname, name);
        Py_DECREF(modname);
        Py_XDECREF(name);
        if (r < 0) {
            return -1;
        }
        if (r > 0) {
            return 0;
        }
    }
    return refuse("call", callable, NULL);
}

/* Call `callable` from restricted code: gated, and lifted when it is a listed C callable. */
PyObject *
_PyRestricted_Vectorcall(PyThreadState *tstate, PyObject *callable, PyObject *const *args, size_t nargsf,
                         PyObject *kwnames)
{
    if (!_PyRestricted_Active(tstate)) {
        return PyObject_Vectorcall(callable, args, nargsf, kwnames);
    }
    int lift = _PyRestricted_CheckCall(callable);
    if (lift < 0) {
        return NULL;
    }
    if (lift == 0) {
        return PyObject_Vectorcall(callable, args, nargsf, kwnames);
    }
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    PyObject *res = PyObject_Vectorcall(callable, args, nargsf, kwnames);
    _PyRestricted_Restore(tstate, saved);
    return res;
}

/* The same for a call with a tuple and a dict (CALL_FUNCTION_EX, f(*a, **k)). */
PyObject *
_PyRestricted_Call(PyThreadState *tstate, PyObject *callable, PyObject *args, PyObject *kwargs)
{
    if (!_PyRestricted_Active(tstate)) {
        return PyObject_Call(callable, args, kwargs);
    }
    int lift = _PyRestricted_CheckCall(callable);
    if (lift < 0) {
        return NULL;
    }
    if (lift == 0) {
        return PyObject_Call(callable, args, kwargs);
    }
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    PyObject *res = PyObject_Call(callable, args, kwargs);
    _PyRestricted_Restore(tstate, saved);
    return res;
}

/* --- registration, from Modules/_restrictedmodule.c (host-only) --- */

static int
ensure(PyObject **slot, PyObject *(*make)(PyObject *))
{
    if (*slot == NULL) {
        *slot = make(NULL);
    }
    return *slot != NULL ? 0 : -1;
}

static PyObject *new_dict(PyObject *unused) { return PyDict_New(); }
static PyObject *new_set(PyObject *unused) { return PySet_New(NULL); }

int
_PyRestricted_AllowKind(PyObject *type, PyObject *names)
{
    if (ensure(&kinds, new_dict) < 0) {
        return -1;
    }
    if (item_read == NULL) {
        item_read = PyUnicode_InternFromString("[]");
        item_write = PyUnicode_InternFromString("[]=");
        call_marker = PyUnicode_InternFromString("()");
        item_lift = PyUnicode_InternFromString("[]^");
        if (item_read == NULL || item_write == NULL || call_marker == NULL || item_lift == NULL) {
            return -1;
        }
    }
    PyObject *fs = PyFrozenSet_New(names);
    if (fs == NULL) {
        return -1;
    }
    int r = PyDict_SetItem(kinds, type, fs);
    Py_DECREF(fs);
    return r;
}

int
_PyRestricted_AllowModule(PyObject *name, PyObject *names)
{
    if (ensure(&modules, new_dict) < 0) {
        return -1;
    }
    PyObject *fs = PyFrozenSet_New(names);
    if (fs == NULL) {
        return -1;
    }
    int r = PyDict_SetItem(modules, name, fs);
    Py_DECREF(fs);
    return r;
}

int
_PyRestricted_AllowCall(PyObject *callable, int lift)
{
    PyObject **slot = lift ? &calls : &calls_unlifted;
    if (ensure(slot, new_set) < 0) {
        return -1;
    }
    return PySet_Add(*slot, callable);
}

void
_PyRestricted_SetException(PyObject *exc)
{
    Py_XSETREF(not_in_dialect, Py_NewRef(exc));
}

void
_PyRestricted_Reset(void)
{
    Py_CLEAR(kinds);
    Py_CLEAR(modules);
    Py_CLEAR(calls);
    Py_CLEAR(calls_unlifted);
}
