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
#include "pycore_tuple.h"         // _PyTuple_FromArray()
#include "pycore_hashtable.h"     // _Py_hashtable_t: the owned set
#include "pycore_setobject.h"     // _PySet_Update()
#include "pycore_pymem.h"         // _PyMem_GetCurrentAllocatorName()

static PyObject *kinds;           // dict: type -> frozenset[str]
static PyObject *modules;         // dict: str -> frozenset[str]
static PyObject *calls;           // dict: callable (or a method's function) -> entry checker or None; called LIFTED
static PyObject *dynamic;         // dict: type -> predicate(obj, name): a name outside the table the kind admits
static PyObject *calls_unlifted;  // set of callables called WITHOUT lifting (pure probes such as _restricted.active)
static PyObject *costs;           // dict: unlifted callable -> cost checker, run before the call (range's length cap)
static PyObject *not_in_dialect;  // the exception type, registered by _restricted
static PyObject *mutators;        // dict: type -> frozenset of method names that write their receiver
static PyObject *fresh;           // set: lifted callables (or a method's function) whose result is a new value
static PyObject *snapshots;       // dict: type -> (snapshot, restore): the undo log for kinds CPython cannot copy

static int is_owned(PyObject *obj);
int _PyRestricted_WriteBarrier(PyObject *obj);
int _PyRestricted_MemoryTripped(void);
static void mark_fresh(PyObject *callable, PyObject *result, PyObject *const *args, Py_ssize_t nargs, PyObject *self);

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

/* Every refusal TRIPS the run: library code that catches it (pandas has `except Exception` fallbacks) cannot turn it
   into a value, because _restricted.run() refuses a tripped run however it ended. */
static void
trip(void)
{
    ((_PyThreadStateImpl *)_PyThreadState_GET())->restricted_tripped = 1;
}

static int
refuse(const char *what, PyObject *obj, PyObject *name)
{
    PyObject *exc = not_in_dialect ? not_in_dialect : PyExc_PermissionError;
    trip();
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
static PyObject *dunder_lift;     // interned "__^": the kind's Python dunders, reached through a type slot, run lifted

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
        if (r == 0 && dynamic != NULL) {
            /* The kind's dynamic rule (a manifest's `dynamic: columns`): never for a dunder (refused above), never for
               a name the table lists (that is the method, as in pandas), and decided by OUR predicate, run lifted,
               never by the object's own __getattr__. 1 = admitted: read it lifted. */
            PyObject *pred;
            int d = PyDict_GetItemRef(dynamic, (PyObject *)Py_TYPE(obj), &pred);
            if (d < 0) {
                return -1;
            }
            if (d > 0) {
                PyThreadState *tstate = _PyThreadState_GET();
                _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
                PyObject *ok = PyObject_CallFunctionObjArgs(pred, obj, name, NULL);
                _PyRestricted_Restore(tstate, saved);
                Py_DECREF(pred);
                if (ok == NULL) {
                    return -1;
                }
                int truth = PyObject_IsTrue(ok);
                Py_DECREF(ok);
                if (truth < 0) {
                    return -1;
                }
                if (truth) {
                    return 1;
                }
            }
        }
    }
    if (r < 0) {
        return -1;
    }
    if (r && !PyModule_Check(obj) && mutators != NULL) {
        /* A method that writes its receiver (list.append, dict.update) is not even READ off a value the cell does
           not own: then no bound mutator of such a value exists, so none can be handed to a C loop (map, sorted's
           key) that would call it past the gate. */
        int m = listed(mutators, (PyObject *)Py_TYPE(obj), name);
        if (m < 0) {
            return -1;
        }
        if (m && !is_owned(obj)) {
            return refuse("a method that writes a value the cell does not own:", obj, name);
        }
        if (m && _PyRestricted_WriteBarrier(obj) < 0) {
            return -1;
        }
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
    if (write && !is_owned(obj)) {
        return refuse("item assignment to a value the cell does not own:", obj, NULL);
    }
    if (write && _PyRestricted_WriteBarrier(obj) < 0) {
        return -1;
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
        int r = PyDict_Contains(calls, key);
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

int
_PyRestricted_IsLiftedLeaf(PyObject *callable)
{
    if (calls == NULL) {
        return 0;
    }
    int r = PyDict_Contains(calls, callable);
    if (r < 0) {
        PyErr_Clear();
        return 1;       /* fail toward the gated path */
    }
    return r;
}

/* Run a lifted leaf's entry checker, if it has one, with the call's arguments: lifted (it is host code), before the
   leaf. It refuses by raising (NotInDialect). The manifest's closed vocabulary and the own-functions-only callback rule
   live there. Returns 0 to proceed, -1 refused. */
static int
run_checker(PyThreadState *tstate, PyObject *table, PyObject *callable, PyObject *args_tuple, PyObject *kwargs)
{
    PyObject *key = PyMethod_Check(callable) ? PyMethod_GET_FUNCTION(callable) : callable;
    PyObject *checker;
    if (table == NULL) {
        return 0;
    }
    int r = PyDict_GetItemRef(table, key, &checker);
    if (r <= 0) {
        return r;
    }
    if (checker == Py_None) {
        Py_DECREF(checker);
        return 0;
    }
    PyObject *self = PyMethod_Check(callable) ? PyMethod_GET_SELF(callable) : Py_None;
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    PyObject *res = PyObject_CallFunctionObjArgs(checker, key, self, args_tuple, kwargs ? kwargs : Py_None, NULL);
    _PyRestricted_Restore(tstate, saved);
    Py_DECREF(checker);
    if (res == NULL) {
        trip();
        return -1;
    }
    Py_DECREF(res);
    return 0;
}

/* Build the tuple and dict a checker sees from a vectorcall's arguments, and run the checker from `table`. */
static int
run_checker_vector(PyThreadState *tstate, PyObject *table, PyObject *callable, PyObject *const *args, size_t nargsf,
                   PyObject *kwnames)
{
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *t = _PyTuple_FromArray(args, nargs);
    PyObject *kw = NULL;
    if (t == NULL) {
        return -1;
    }
    if (kwnames != NULL && PyTuple_GET_SIZE(kwnames) > 0) {
        kw = PyDict_New();
        for (Py_ssize_t i = 0; kw != NULL && i < PyTuple_GET_SIZE(kwnames); i++) {
            if (PyDict_SetItem(kw, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]) < 0) {
                Py_CLEAR(kw);
            }
        }
        if (kw == NULL) {
            Py_DECREF(t);
            return -1;
        }
    }
    int c = run_checker(tstate, table, callable, t, kw);
    Py_DECREF(t);
    Py_XDECREF(kw);
    return c;
}

/* An unlifted callable with a cost checker (a C loop the budget cannot see, bounded by its arguments before it runs). */
static int
has_cost(PyObject *callable)
{
    if (costs == NULL) {
        return 0;
    }
    int r = PyDict_Contains(costs, callable);
    if (r < 0) {
        PyErr_Clear();
        return 1;       /* fail toward running the check, which then reports the error */
    }
    return r;
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
        if (has_cost(callable) && run_checker_vector(tstate, costs, callable, args, nargsf, kwnames) < 0) {
            return NULL;
        }
        return PyObject_Vectorcall(callable, args, nargsf, kwnames);
    }
    /* The checker sees the arguments as a tuple and a dict. */
    if (run_checker_vector(tstate, calls, callable, args, nargsf, kwnames) < 0) {
        return NULL;
    }
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    PyObject *res = PyObject_Vectorcall(callable, args, nargsf, kwnames);
    _PyRestricted_Restore(tstate, saved);
    mark_fresh(callable, res, args, PyVectorcall_NARGS(nargsf) + (kwnames ? PyTuple_GET_SIZE(kwnames) : 0),
               PyMethod_Check(callable) ? PyMethod_GET_SELF(callable) : NULL);
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
        if (has_cost(callable) && run_checker(tstate, costs, callable, args, kwargs) < 0) {
            return NULL;
        }
        return PyObject_Call(callable, args, kwargs);
    }
    if (run_checker(tstate, calls, callable, args, kwargs) < 0) {
        return NULL;
    }
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    PyObject *res = PyObject_Call(callable, args, kwargs);
    _PyRestricted_Restore(tstate, saved);
    if (res != NULL && PyTuple_Check(args)) {
        /* Keyword values are not compared here: a leaf handing back a keyword argument is left fresh-marked only
           if the manifest says fresh, and the corpus has no such call (f(*a, **k) is rare). */
        mark_fresh(callable, res, &PyTuple_GET_ITEM(args, 0), PyTuple_GET_SIZE(args),
                   PyMethod_Check(callable) ? PyMethod_GET_SELF(callable) : NULL);
    }
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
        dunder_lift = PyUnicode_InternFromString("__^");
        if (item_read == NULL || item_write == NULL || call_marker == NULL || item_lift == NULL
            || dunder_lift == NULL)
        {
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
_PyRestricted_AllowCall(PyObject *callable, int lift, PyObject *checker)
{
    if (!lift) {
        if (ensure(&calls_unlifted, new_set) < 0) {
            return -1;
        }
        if (checker != NULL && checker != Py_None) {
            if (ensure(&costs, new_dict) < 0 || PyDict_SetItem(costs, callable, checker) < 0) {
                return -1;
            }
        }
        return PySet_Add(calls_unlifted, callable);
    }
    if (ensure(&calls, new_dict) < 0) {
        return -1;
    }
    return PyDict_SetItem(calls, callable, checker ? checker : Py_None);
}

int
_PyRestricted_AllowDynamic(PyObject *type, PyObject *predicate)
{
    if (ensure(&dynamic, new_dict) < 0) {
        return -1;
    }
    return PyDict_SetItem(dynamic, type, predicate);
}

/* Read an attribute the kind's DYNAMIC rule admitted (a DataFrame column), lifted: the object's own lookup is host
   code (pandas' __getattr__). 0 with *result set, -1 on error. */
int
_PyRestricted_ReadAttrLifted(PyObject *obj, PyObject *name, PyObject **result)
{
    PyThreadState *tstate = _PyThreadState_GET();
    _PyRestrictedSaved saved = _PyRestricted_Lift(tstate);
    *result = PyObject_GetAttr(obj, name);
    _PyRestricted_Restore(tstate, saved);
    return *result == NULL ? -1 : 0;
}

/* A CO_RESTRICTED frame entered while no run is active: the cell's own function, called by host code after its run
   (a later approved cell, exec() of its code, a C library handed it as a callback). Refused rather than run, so code
   the gate never saw runs nowhere outside it; the text tells the caller what to do instead. */
int
_PyRestricted_RefuseOutside(struct _PyInterpreterFrame *frame)
{
    PyObject *exc = not_in_dialect ? not_in_dialect : PyExc_PermissionError;
    /* Named as the cell wrote it: without the wrapper's "_user.<locals>." (restricted.compile_cell). */
    PyObject *name = _PyFrame_GetCode(frame)->co_qualname, *shown = NULL;
    PyObject *prefix = PyUnicode_FromString("_user.<locals>.");
    if (prefix != NULL && PyUnicode_Tailmatch(name, prefix, 0, PY_SSIZE_T_MAX, -1) > 0) {
        shown = PyUnicode_Substring(name, PyUnicode_GET_LENGTH(prefix), PyUnicode_GET_LENGTH(name));
    }
    Py_XDECREF(prefix);
    PyErr_Clear();   /* a failure above only loses the shorter name */
    PyErr_Format(exc, "not in dialect: %U() was defined in a cell that ran inside the read-only dialect, unasked, "
                 "and runs only there; define it again in this cell to call it here", shown ? shown : name);
    Py_XDECREF(shown);
    return -1;
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
    Py_CLEAR(costs);
    Py_CLEAR(mutators);
    Py_CLEAR(fresh);
    Py_CLEAR(snapshots);
    Py_CLEAR(dynamic);
}

/* --- the step budget and the depth (Python/ceval_gil.c calls this from _Py_HandlePending) --- */

/* One step: charged wherever CPython already checks the eval breaker, which is every backward jump and every call,
   i.e. where loops and recursion are. Lifted library Python is charged too: the budget measures interpreter work in
   the run, whoever's code it is. Once exhausted the budget stays exhausted, so every later check refuses again and a
   library `except Exception` that swallowed one refusal meets the next at its own next loop or call. The DEPTH is
   judged only at a CO_RESTRICTED frame (a program's own def or lambda), counted in Python frames above where the run
   began, so a lifted leaf's internals may nest as CPython allows while the program's recursion is bounded. */
int
_PyRestricted_Tick(PyThreadState *tstate)
{
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    PyObject *exc = not_in_dialect ? not_in_dialect : PyExc_PermissionError;
    if (_PyRestricted_MemoryTripped()) {
        trip();
        PyErr_Format(exc, "not in dialect: the cell's memory cap ran out");
        return -1;
    }
    if (--ts->restricted_steps < 0) {
        ts->restricted_steps = -1;      /* stays exhausted, and cannot wrap */
        trip();
        PyErr_Format(exc, "not in dialect: the step budget (%lld) ran out", ts->restricted_budget);
        return -1;
    }
    _PyInterpreterFrame *f = tstate->current_frame;
    if (f != NULL && f->owner != FRAME_OWNED_BY_INTERPRETER) {
        PyObject *exec = PyStackRef_AsPyObjectBorrow(f->f_executable);
        if (PyCode_Check(exec) && (((PyCodeObject *)exec)->co_flags & CO_RESTRICTED)) {
            int depth = ts->restricted_depth_base - tstate->py_recursion_remaining;
            if (depth > ts->restricted_depth_limit) {
                trip();
                PyErr_Format(exc, "not in dialect: recursion past the depth (%d frames)", ts->restricted_depth_limit);
                return -1;
            }
        }
    }
    return 0;
}


/* --- ownership: the owned side set (slice 5) ---

   A value is the cell's OWN when the dialect made it: a list, dict, set or bytearray created while the gate is active
   (their constructors always copy, so a new one never aliases anything), or a lifted leaf's result the manifest marks
   FRESH. Nothing else is: not a view (a column Series, df.values, a reshape), not a library's internal object, not an
   injected value. Writes (item assignment, an in-place operator, reading a mutating method) need an owned target.

   The set is keyed by pointer and kept in step with the heap through CPython's reference tracer (PyRefTracer_SetTracer),
   which sees every creation and every deallocation, so the object layout is unchanged and a freed address that is
   reused is never owned by accident. If another tracer replaces ours, every ownership is FORGOTTEN at the next run
   (owned_reset): unowned is the safe direction. */

static _Py_hashtable_t *owned;      /* value: the run epoch the object became owned in */
static uintptr_t owned_epoch = 1;   /* the current outermost run's epoch (never 0, which reads as "absent") */
#define MAX_TRACKED 32
static PyTypeObject *tracked[MAX_TRACKED];      /* types whose instances may be in the set */
static int ntracked;

static int
is_tracked(PyTypeObject *tp)
{
    for (int i = 0; i < ntracked; i++) {
        if (tracked[i] == tp) {
            return 1;
        }
    }
    return 0;
}

static int
track(PyTypeObject *tp)
{
    if (is_tracked(tp)) {
        return 0;
    }
    if (ntracked == MAX_TRACKED) {
        return -1;
    }
    tracked[ntracked++] = tp;
    return 0;
}

/* The types whose new instance, made under the gate, is owned. Not ndarray, DataFrame or Series: a new one may be a
   view of existing data, so those are owned only as a FRESH leaf result. */
static int
owned_on_create(PyTypeObject *tp)
{
    return tp == &PyList_Type || tp == &PyDict_Type || tp == &PySet_Type || tp == &PyByteArray_Type;
}

/* Generators (and coroutines) created inside a LIFTED extent: library code's, so resuming one from restricted code
   (FOR_ITER over pandas' string-array __iter__) runs it lifted again. One created under the gate (a tier-3 library
   generator function called from restricted code) is not here, and stays gated when resumed. */
static _Py_hashtable_t *lifted_gens;

static inline int
is_generator_type(PyTypeObject *tp)
{
    return tp == &PyGen_Type || tp == &PyCoro_Type || tp == &PyAsyncGen_Type;
}

static int
ref_tracer(PyObject *op, PyRefTracerEvent event, void *data)
{
    PyTypeObject *tp = Py_TYPE(op);
    if (event == PyRefTracer_CREATE) {
        if (owned_on_create(tp)) {
            PyThreadState *tstate = _PyThreadState_GET();
            if (tstate != NULL && _PyRestricted_Active(tstate)) {
                (void)_Py_hashtable_set(owned, op, (void *)owned_epoch);  /* on failure: simply not owned */
            }
        }
        else if (is_generator_type(tp)) {
            PyThreadState *tstate = _PyThreadState_GET();
            if (tstate != NULL && ((_PyThreadStateImpl *)tstate)->restricted_lifted
                && !_PyRestricted_Active(tstate))
            {
                (void)_Py_hashtable_set(lifted_gens, op, op);   /* on failure it stays gated: the safe side */
            }
        }
    }
    else {
        /* Both, not either: a generator made in a lifted extent and returned by a FRESH leaf is in both tables, and
           a stale lifted_gens entry would resume a later generator at that address lifted. */
        if (ntracked && is_tracked(tp)) {
            (void)_Py_hashtable_steal(owned, op);
        }
        if (is_generator_type(tp) && _Py_hashtable_len(lifted_gens)) {
            (void)_Py_hashtable_steal(lifted_gens, op);
        }
    }
    return 0;
}

int
_PyRestricted_IsLiftedGen(PyObject *gen)
{
    return lifted_gens != NULL && _Py_hashtable_get(lifted_gens, gen) != NULL;
}

static int
owned_reset(void)
{
    void *data;
    if (owned != NULL && PyRefTracer_GetTracer(&data) == ref_tracer) {
        return 0;
    }
    if (owned == NULL) {
        owned = _Py_hashtable_new(_Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
        lifted_gens = _Py_hashtable_new(_Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
        if (owned == NULL || lifted_gens == NULL) {
            PyErr_NoMemory();
            return -1;
        }
    }
    _Py_hashtable_clear(owned);
    _Py_hashtable_clear(lifted_gens);
    ntracked = 0;
    if (track(&PyList_Type) < 0 || track(&PyDict_Type) < 0 || track(&PySet_Type) < 0
        || track(&PyByteArray_Type) < 0)
    {
        return -1;
    }
    return PyRefTracer_SetTracer(ref_tracer, NULL);
}

static int
is_owned(PyObject *obj)
{
    return owned != NULL && _Py_hashtable_get(owned, obj) != NULL;
}

/* Called by _restricted.run() before a run: install the tracer, or forget every ownership if ours was replaced. */
int
_PyRestricted_OwnedReady(void)
{
    return owned_reset();
}

int
_PyRestricted_IsOwned(PyObject *obj)
{
    return is_owned(obj);
}

/* Test hook: the sizes of the owned set and lifted_gens, so a test can see an entry outlive its object. */
void
_PyRestricted_TableSizes(Py_ssize_t *n_owned, Py_ssize_t *n_lifted_gens)
{
    *n_owned = owned == NULL ? 0 : (Py_ssize_t)_Py_hashtable_len(owned);
    *n_lifted_gens = lifted_gens == NULL ? 0 : (Py_ssize_t)_Py_hashtable_len(lifted_gens);
}

/* Mark a lifted leaf's FRESH result owned. Refused as a marking (left unowned) when it is the receiver or an
   argument: a leaf that hands back what it was given (df.pipe(lambda d: d)) made nothing new. */
static void
mark_fresh(PyObject *callable, PyObject *result, PyObject *const *args, Py_ssize_t nargs, PyObject *self)
{
    if (fresh == NULL || result == NULL || owned == NULL) {
        return;
    }
    PyObject *key = PyMethod_Check(callable) ? PyMethod_GET_FUNCTION(callable) : callable;
    int r = PySet_Contains(fresh, key);
    if (r <= 0) {
        if (r < 0) {
            PyErr_Clear();
        }
        return;
    }
    if (result == self) {
        return;
    }
    for (Py_ssize_t i = 0; i < nargs; i++) {
        if (args[i] == result) {
            return;
        }
    }
    if (track(Py_TYPE(result)) == 0) {
        (void)_Py_hashtable_set(owned, result, (void *)owned_epoch);
    }
}

/* An in-place operator on `v` that writes it (the type has the in-place slot): only on an owned value. */
int
_PyRestricted_CheckInPlace(PyObject *v)
{
    if (is_owned(v)) {
        return _PyRestricted_WriteBarrier(v);
    }
    return refuse("an in-place operator on a value the cell does not own:", v, NULL);
}

int
_PyRestricted_AllowMutators(PyObject *type, PyObject *names)
{
    if (ensure(&mutators, new_dict) < 0) {
        return -1;
    }
    PyObject *fs = PyFrozenSet_New(names);
    if (fs == NULL) {
        return -1;
    }
    int r = PyDict_SetItem(mutators, type, fs);
    Py_DECREF(fs);
    return r;
}

int
_PyRestricted_AllowFresh(PyObject *callable)
{
    if (ensure(&fresh, new_set) < 0) {
        return -1;
    }
    return PySet_Add(fresh, callable);
}

/* --- memory: a byte cap per run (slice 5) ---

   The native form of the JS dialect's size caps. Under pymalloc, CPython's RAW allocator domain receives every block
   over 512 bytes (pymalloc hands those to PyMem_RawMalloc) and the arena allocator every arena of small objects, so
   wrapping those two counts the heap a run adds, net of what it frees, at almost no cost per small object. Without
   pymalloc (Emscripten and WASI build --without-pymalloc, so this is Pyodide's case; also PYTHONMALLOC=malloc or
   mimalloc) the MEM and OBJ domains call malloc directly and never reach RAW, so they are wrapped too, each block
   counted, for the outermost run's extent only: outside a run they allocate with nothing in between. A request that would take the run past its cap FAILS before anything is allocated (so "a" * 10**11 is
   refused in microseconds instead of touching 100 GB), and trips the run, so the MemoryError the caller raises ends as
   NotInDialect however it is handled. The wrappers are installed once, as hooks that call the previous allocator (the
   documented way to hook after initialization), and count only on the running thread while a run is in progress.

   A request CPython refuses with its own overflow guard never reaches an allocator: on a 32-bit build "a" * 10**11
   fails converting the count to Py_ssize_t, an OverflowError and not the cap. _PyRestricted_RepeatAdmit judges a
   sequence repeat's size before that guard (Objects/abstract.c, sequence_repeat).

   Not counted: memory a C library takes from malloc directly, which includes numpy's array data (PyDataMem). A numpy
   leaf's cost and size are its manifest's (tier 1 bounds over shape), not this cap's. */

static PyMemAllocatorEx raw_orig, mem_orig, obj_orig;     /* each wrapper's ctx is the domain's previous allocator */
static PyObjectArenaAllocator arena_orig;
static int mem_installed;
static int mem_counting;
static unsigned long mem_thread;
static long long mem_live, mem_peak, mem_limit;
static int mem_tripped;
static _Py_hashtable_t *mem_blocks;         /* block -> size, for the blocks a run allocated through a wrapped domain */

static void *ht_malloc(size_t n) { return raw_orig.malloc(raw_orig.ctx, n); }
static void ht_free(void *p) { raw_orig.free(raw_orig.ctx, p); }

static inline int
mem_mine(void)
{
    return mem_counting && PyThread_get_thread_ident() == mem_thread;
}

/* May the run grow by `n` bytes? If not, the run is tripped (and stays so: the next periodic check refuses, and so
   does run() at the end). Only a request past the cap fails, so the refusal itself still has room to be raised. */
static int
mem_admit(long long n)
{
    if (n > 0 && mem_live + n > mem_limit) {
        mem_tripped = 1;
        return 0;
    }
    return 1;
}

static void
mem_grew(long long n)
{
    mem_live += n;
    if (mem_live > mem_peak) {
        mem_peak = mem_live;
    }
}

static void *
mem_record(void *p, size_t n)
{
    if (p != NULL && _Py_hashtable_set(mem_blocks, p, (void *)(uintptr_t)(n ? n : 1)) == 0) {
        mem_grew((long long)n);
    }
    return p;
}

static long long
mem_forget(void *p)
{
    if (p == NULL) {
        return 0;
    }
    return (long long)(uintptr_t)_Py_hashtable_steal(mem_blocks, p);
}

static void *
counted_malloc(void *ctx, size_t n)
{
    PyMemAllocatorEx *o = ctx;
    if (!mem_mine()) {
        return o->malloc(o->ctx, n);
    }
    if (!mem_admit((long long)n)) {
        return NULL;
    }
    return mem_record(o->malloc(o->ctx, n), n);
}

static void *
counted_calloc(void *ctx, size_t nelem, size_t elsize)
{
    PyMemAllocatorEx *o = ctx;
    if (!mem_mine()) {
        return o->calloc(o->ctx, nelem, elsize);
    }
    if (elsize != 0 && nelem > (size_t)PY_SSIZE_T_MAX / elsize) {
        mem_tripped = 1;
        return NULL;
    }
    size_t n = nelem * elsize;
    if (!mem_admit((long long)n)) {
        return NULL;
    }
    return mem_record(o->calloc(o->ctx, nelem, elsize), n);
}

static void *
counted_realloc(void *ctx, void *ptr, size_t n)
{
    PyMemAllocatorEx *o = ctx;
    if (!mem_mine()) {
        return o->realloc(o->ctx, ptr, n);
    }
    long long old = mem_forget(ptr);          /* 0 for a block from before the run: then all of n counts */
    if (!mem_admit((long long)n - old)) {
        if (old) {
            (void)_Py_hashtable_set(mem_blocks, ptr, (void *)(uintptr_t)old);
        }
        return NULL;
    }
    void *p = o->realloc(o->ctx, ptr, n);
    if (p == NULL) {
        if (old) {
            (void)_Py_hashtable_set(mem_blocks, ptr, (void *)(uintptr_t)old);
        }
        return NULL;
    }
    mem_live -= old;
    return mem_record(p, n);
}

static void
counted_free(void *ctx, void *ptr)
{
    PyMemAllocatorEx *o = ctx;
    if (mem_mine()) {
        mem_live -= mem_forget(ptr);
    }
    o->free(o->ctx, ptr);
}

static void *
arena_alloc(void *ctx, size_t size)
{
    if (!mem_mine()) {
        return arena_orig.alloc(arena_orig.ctx, size);
    }
    if (!mem_admit((long long)size)) {
        return NULL;
    }
    void *p = arena_orig.alloc(arena_orig.ctx, size);
    if (p != NULL) {
        mem_grew((long long)size);
    }
    return p;
}

static void
arena_free(void *ctx, void *ptr, size_t size)
{
    if (mem_mine()) {
        mem_live -= (long long)size;
    }
    arena_orig.free(arena_orig.ctx, ptr, size);
}

/* Whether pymalloc sits under MEM and OBJ. The name is read first; a hook installed before ours (tracemalloc) hides
   it, and then the allocator chosen at startup (PYTHONMALLOC, -X dev, an embedder's PyPreConfig) decides. */
static int
pymalloc_underneath(void)
{
    const char *name = _PyMem_GetCurrentAllocatorName();
    if (name != NULL) {
        return strcmp(name, "pymalloc") == 0 || strcmp(name, "pymalloc_debug") == 0;
    }
#if defined(WITH_PYMALLOC) && !defined(Py_GIL_DISABLED)
    switch (_PyRuntime.preconfig.allocator) {
    case PYMEM_ALLOCATOR_NOT_SET:
    case PYMEM_ALLOCATOR_DEFAULT:
    case PYMEM_ALLOCATOR_DEBUG:
    case PYMEM_ALLOCATOR_PYMALLOC:
    case PYMEM_ALLOCATOR_PYMALLOC_DEBUG:
        return 1;
    default:
        return 0;
    }
#elif defined(WITH_PYMALLOC)
    PyMemAllocatorName a = _PyRuntime.preconfig.allocator;
    return a == PYMEM_ALLOCATOR_PYMALLOC || a == PYMEM_ALLOCATOR_PYMALLOC_DEBUG;
#else
    return 0;
#endif
}

/* Is `cur` our wrapper over `orig`? */
static int
mem_is_ours(const PyMemAllocatorEx *cur, PyMemAllocatorEx *orig)
{
    return cur->malloc == counted_malloc && cur->ctx == orig;
}

static int mem_wrap_domains;        /* decided once: does this allocator need MEM and OBJ wrapped (not pymalloc)? */
static int mem_in_chain[2];         /* is our wrapper in the MEM / OBJ domain's chain, on top or under another hook? */

static PyMemAllocatorEx *
domain_orig(int i)
{
    return i == 0 ? &mem_orig : &obj_orig;
}

/* Wrap MEM and OBJ for the outermost run's extent, so that code outside a run allocates through the original allocator
   with no call in between. Both domains need the GIL, so the swap cannot race another thread's allocation. */
static void
mem_domains_wrap(void)
{
    if (!mem_wrap_domains) {
        return;
    }
    for (int i = 0; i < 2; i++) {
        PyMemAllocatorDomain d = i == 0 ? PYMEM_DOMAIN_MEM : PYMEM_DOMAIN_OBJ;
        if (mem_in_chain[i]) {
            continue;                   /* still under a hook stacked over it during an earlier run: it counts there */
        }
        PyMem_GetAllocator(d, domain_orig(i));
        PyMemAllocatorEx ours = {domain_orig(i), counted_malloc, counted_calloc, counted_realloc, counted_free};
        PyMem_SetAllocator(d, &ours);
        mem_in_chain[i] = 1;
    }
}

/* Put the originals back, but only where ours is still on top. A hook stacked over ours during the run (tracemalloc
   started by host code) keeps ours as its inner allocator; restoring past it would drop that hook, and its own restore
   later would put ours back for good, to be wrapped again by the next run and count every block twice. So a buried
   wrapper is left in place, a pass-through while no run is counting, and taken out by the first run that ends with
   it back on top. */
static void
mem_domains_unwrap(void)
{
#ifdef Py_GIL_DISABLED
    /* Without a GIL another thread can be inside MEM or OBJ while the allocator is swapped, so a free-threaded build
       keeps the wrappers installed from the first run on, a pass-through outside a run. */
    return;
#endif
    for (int i = 0; i < 2; i++) {
        PyMemAllocatorDomain d = i == 0 ? PYMEM_DOMAIN_MEM : PYMEM_DOMAIN_OBJ;
        PyMemAllocatorEx cur;
        PyMem_GetAllocator(d, &cur);
        if (mem_in_chain[i] && mem_is_ours(&cur, domain_orig(i))) {
            PyMem_SetAllocator(d, domain_orig(i));
            mem_in_chain[i] = 0;
        }
    }
}

static int
mem_install(void)
{
    if (mem_installed) {
        return 0;
    }
    /* Only pymalloc routes MEM and OBJ through RAW and the arenas; wrapping them under it would count a block twice.
       Read before RAW is wrapped: after, the name matches no known allocator and reads as NULL. */
    mem_wrap_domains = !pymalloc_underneath();
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &raw_orig);
    PyObject_GetArenaAllocator(&arena_orig);
    _Py_hashtable_allocator_t a = {ht_malloc, ht_free};
    mem_blocks = _Py_hashtable_new_full(_Py_hashtable_hash_ptr, _Py_hashtable_compare_direct, NULL, NULL, &a);
    if (mem_blocks == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    PyMemAllocatorEx raw = {&raw_orig, counted_malloc, counted_calloc, counted_realloc, counted_free};
    PyObjectArenaAllocator arena = {NULL, arena_alloc, arena_free};
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &raw);
    PyObject_SetArenaAllocator(&arena);
    mem_installed = 1;
    return 0;
}

/* Test hook: whether MEM and OBJ are wrapped for this allocator at all, and whether ours is on top of each now. */
void
_PyRestricted_MemoryWrapped(int *wraps, int *mem_on_top, int *obj_on_top)
{
    PyMemAllocatorEx cur;
    *wraps = mem_installed ? mem_wrap_domains : -1;
    PyMem_GetAllocator(PYMEM_DOMAIN_MEM, &cur);
    *mem_on_top = mem_is_ours(&cur, &mem_orig);
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &cur);
    *obj_on_top = mem_is_ours(&cur, &obj_orig);
}

/* Start counting for a run on this thread, capped at `limit` bytes. Returns 1 when this call started the count (the
   outermost run), 0 when a run is already counting (a nested run shares the outer cap), -1 on error. */
int
_PyRestricted_MemoryBegin(long long limit)
{
    if (mem_install() < 0) {
        return -1;
    }
    if (mem_counting) {
        return 0;
    }
    _Py_hashtable_clear(mem_blocks);
    mem_live = mem_peak = 0;
    mem_limit = limit;
    mem_tripped = 0;
    mem_thread = PyThread_get_thread_ident();
    mem_domains_wrap();
    mem_counting = 1;
    return 1;
}

/* Stop counting (the outermost run). Returns whether the cap was hit; *peak is the most the run held at once. */
int
_PyRestricted_MemoryEnd(long long *peak)
{
    mem_counting = 0;
    mem_domains_unwrap();
    _Py_hashtable_clear(mem_blocks);
    *peak = mem_peak;
    return mem_tripped;
}

int
_PyRestricted_MemoryTripped(void)
{
    return mem_counting && mem_tripped;
}

/* May `seq * n` be made? A repeat of a str, bytes, bytearray, list or tuple asks for at least len * n items of its
   narrowest element, judged here against the cap before CPython's own overflow guards (a count past Py_ssize_t, a
   product past it), which refuse without asking an allocator and so never trip the run. The estimate never exceeds
   what the allocation would take, so it refuses nothing the allocator would admit. 0 = go on, -1 = tripped. */
int
_PyRestricted_RepeatAdmit(PyObject *seq, PyObject *n)
{
    int sign;
    if (!mem_mine() || !PyLong_Check(n) || PyLong_GetSign(n, &sign) < 0 || sign <= 0) {
        return 0;
    }
    Py_ssize_t len;
    double item;
    if (PyUnicode_Check(seq)) {
        len = PyUnicode_GET_LENGTH(seq);
        item = PyUnicode_KIND(seq);
    }
    else if (PyBytes_Check(seq) || PyByteArray_Check(seq)) {
        len = Py_SIZE(seq);
        item = 1;
    }
    else if (PyList_Check(seq) || PyTuple_Check(seq)) {
        len = Py_SIZE(seq);
        item = sizeof(PyObject *);
    }
    else {
        return 0;
    }
    if (len == 0) {
        return 0;
    }
    double count = PyLong_AsDouble(n);
    if (count == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();                      /* past a double: past any cap */
        count = Py_HUGE_VAL;
    }
    if ((double)mem_live + (double)len * item * count > (double)mem_limit) {
        mem_tripped = 1;
        return -1;
    }
    return 0;
}


/* --- what crosses into a lifted extent through the item protocol and the type slots --- */

/* The callback rule for arguments that enter a lifted extent other than through a leaf's entry checker: an item key or
   value under "[]^" (pandas calls a callable key: df[f] is f(df)), and the operands of a lifted dunder. A callable
   must be the program's own function (CO_RESTRICTED); a class is data. Plain containers are walked, a few levels deep,
   as the entry checker walks them. 0 = fine, -1 = refused. */
static int
crossing_ok(PyObject *x, int depth)
{
    if (PyFunction_Check(x)) {
        PyCodeObject *co = (PyCodeObject *)PyFunction_GET_CODE(x);
        if (co->co_flags & CO_RESTRICTED) {
            return 0;
        }
        return refuse("handing a lifted extent a callable that is not the program's own function:", x, NULL);
    }
    if (PyType_Check(x)) {
        return 0;
    }
    if (PyList_CheckExact(x) || PyTuple_CheckExact(x)) {
        if (depth >= 4) {
            return 0;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(x);
        for (Py_ssize_t i = 0; i < n; i++) {
            /* Borrowed: a list item could be released by a concurrent write, but nothing runs between these reads. */
            if (crossing_ok(PySequence_Fast_GET_ITEM(x, i), depth + 1) < 0) {
                return -1;
            }
        }
        return 0;
    }
    if (PyDict_CheckExact(x)) {
        if (depth >= 4) {
            return 0;
        }
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(x, &pos, &k, &v)) {
            if (crossing_ok(k, depth + 1) < 0 || crossing_ok(v, depth + 1) < 0) {
                return -1;
            }
        }
        return 0;
    }
    if (PyCallable_Check(x)) {
        return refuse("handing a lifted extent a callable that is not the program's own function:", x, NULL);
    }
    return 0;
}

int
_PyRestricted_CheckCrossing(PyObject *const *objs, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++) {
        if (objs[i] != NULL && crossing_ok(objs[i], 0) < 0) {
            return -1;
        }
    }
    return 0;
}

/* Does `self`'s kind list "__^": its Python dunders (pandas' __len__, __add__, __eq__, __iter__), reached through a
   type slot from restricted code, run LIFTED, as its listed methods do. Each such slot is reached only after the
   gate's own barrier for the operation (the item read or write, the in-place check, the attribute table), so lifting
   the dunder lifts nothing the gate has not already admitted. */
int
_PyRestricted_LiftsDunders(PyObject *self)
{
    if (kinds == NULL || dunder_lift == NULL) {
        return 0;
    }
    int r = listed(kinds, (PyObject *)Py_TYPE(self), dunder_lift);
    if (r < 0) {
        PyErr_Clear();
        return 0;       /* fail toward gated */
    }
    return r;
}


/* --- rollback: the undo log at the write barrier (slice 6) ---

   The first write a run makes to an owned value that is OLDER than the run (an earlier cell's list, a copy an earlier
   cell made) records a snapshot of that value's whole contents; a value the run itself made needs none, since nothing
   outside the run can see it. A refused run replays every snapshot, so the namespace and every value it reaches are
   as they were before the cell. Whole-object snapshots, once per value per run, rather than one entry per write: a C
   method that rewrites a container in one call (list.sort, dict.update) is covered the same way as an item write.
   Values the barrier admits but CPython cannot copy (a pandas frame, a numpy array) use a SNAPSHOT pair the host
   registers per type (allow_snapshot); a write to an owned value of a kind with neither is refused.

   The barrier sits where ownership is checked: an item write or delete, an in-place operator, reading a mutating
   method, and a lifted leaf's write check (_restricted.write_barrier from the entry checker). */

static PyObject *undo;              /* list of (obj, state, restore-or-None), in write order */
static _Py_hashtable_t *undo_seen;  /* obj -> 1: already snapshotted in this run */
static int undo_depth;              /* nesting of run(): only the outermost run owns the log */

static uintptr_t
owned_since(PyObject *obj)
{
    return owned == NULL ? 0 : (uintptr_t)_Py_hashtable_get(owned, obj);
}

int
_PyRestricted_WriteBarrier(PyObject *obj)
{
    if (undo == NULL || undo_depth == 0) {
        return 0;                   /* no run: nothing to roll back to */
    }
    if (owned_since(obj) == owned_epoch || _Py_hashtable_get(undo_seen, obj) != NULL) {
        return 0;
    }
    PyThreadState *tstate = _PyThreadState_GET();
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    /* Copy with the gate fully off: the copy is host bookkeeping, never the cell's value. */
    _PyRestrictedSaved saved = {ts->restricted, ts->restricted_lifted, ts->restricted_barrier};
    ts->restricted = 0;
    ts->restricted_lifted = 0;
    PyObject *state = NULL, *restore = Py_None;
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *pair = NULL;
    if (tp == &PyList_Type) {
        state = PyList_GetSlice(obj, 0, PY_SSIZE_T_MAX);
    }
    else if (tp == &PyDict_Type) {
        state = PyDict_Copy(obj);
    }
    else if (tp == &PySet_Type) {
        state = PyFrozenSet_New(obj);
    }
    else if (tp == &PyByteArray_Type) {
        state = PyBytes_FromStringAndSize(PyByteArray_AS_STRING(obj), PyByteArray_GET_SIZE(obj));
    }
    else if (snapshots != NULL && PyDict_GetItemRef(snapshots, (PyObject *)tp, &pair) > 0) {
        state = PyObject_CallOneArg(PyTuple_GET_ITEM(pair, 0), obj);
        restore = PyTuple_GET_ITEM(pair, 1);
    }
    else if (!PyErr_Occurred()) {
        _PyRestricted_Restore(tstate, saved);
        Py_XDECREF(pair);
        return refuse("a write that cannot be undone (no snapshot for the kind):", obj, NULL);
    }
    _PyRestricted_Restore(tstate, saved);
    PyObject *entry = state == NULL ? NULL : PyTuple_Pack(3, obj, state, restore);
    Py_XDECREF(state);
    Py_XDECREF(pair);
    if (entry == NULL || PyList_Append(undo, entry) < 0 || _Py_hashtable_set(undo_seen, obj, obj) < 0) {
        Py_XDECREF(entry);
        if (!PyErr_Occurred()) {
            PyErr_NoMemory();
        }
        trip();     /* a write the run could not log must not happen */
        return -1;
    }
    Py_DECREF(entry);
    return 0;
}

/* Begin a run's log: 1 for the outermost run (a new epoch, an empty log), 0 for a nested one. */
int
_PyRestricted_UndoBegin(void)
{
    if (undo_depth++ > 0) {
        return 0;
    }
    if (undo == NULL) {
        undo = PyList_New(0);
        undo_seen = _Py_hashtable_new(_Py_hashtable_hash_ptr, _Py_hashtable_compare_direct);
        if (undo == NULL || undo_seen == NULL) {
            undo_depth--;
            PyErr_NoMemory();
            return -1;
        }
    }
    owned_epoch++;
    return 1;
}

static int
replay_one(PyObject *entry)
{
    PyObject *obj = PyTuple_GET_ITEM(entry, 0);
    PyObject *state = PyTuple_GET_ITEM(entry, 1);
    PyObject *restore = PyTuple_GET_ITEM(entry, 2);
    PyTypeObject *tp = Py_TYPE(obj);
    if (restore != Py_None) {
        PyObject *r = PyObject_CallFunctionObjArgs(restore, obj, state, NULL);
        Py_XDECREF(r);
        return r == NULL ? -1 : 0;
    }
    if (tp == &PyList_Type || tp == &PyByteArray_Type) {
        return PySequence_SetSlice(obj, 0, PY_SSIZE_T_MAX, state);
    }
    if (tp == &PyDict_Type) {
        PyDict_Clear(obj);
        return PyDict_Update(obj, state);
    }
    if (tp == &PySet_Type) {
        return PySet_Clear(obj) < 0 ? -1 : _PySet_Update(obj, state);
    }
    return 0;
}

/* End a run's log, outermost only, with the gate already restored to the caller's state. `refused`: replay every
   snapshot, newest first. Returns 0, or -1 when a replay failed (the caller reports it; the run is refused anyway). */
int
_PyRestricted_UndoEnd(int refused)
{
    if (--undo_depth > 0) {
        return 0;
    }
    int rc = 0;
    if (refused) {
        PyObject *exc = PyErr_GetRaisedException();
        for (Py_ssize_t i = PyList_GET_SIZE(undo) - 1; i >= 0; i--) {
            if (replay_one(PyList_GET_ITEM(undo, i)) < 0) {
                PyErr_Clear();
                rc = -1;
            }
        }
        PyErr_SetRaisedException(exc);
    }
    Py_SETREF(undo, PyList_New(0));
    _Py_hashtable_clear(undo_seen);
    owned_epoch++;      /* what this run made is now older than the next run */
    if (undo == NULL) {
        PyErr_Clear();
        return -1;
    }
    return rc;
}

int
_PyRestricted_AllowSnapshot(PyObject *type, PyObject *snapshot, PyObject *restore)
{
    if (ensure(&snapshots, new_dict) < 0) {
        return -1;
    }
    PyObject *pair = PyTuple_Pack(2, snapshot, restore);
    if (pair == NULL) {
        return -1;
    }
    int r = PyDict_SetItem(snapshots, type, pair);
    Py_DECREF(pair);
    return r;
}
