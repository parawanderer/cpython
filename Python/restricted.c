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

static PyObject *kinds;           // dict: type -> frozenset[str]
static PyObject *modules;         // dict: str -> frozenset[str]
static PyObject *calls;           // dict: callable (or a method's function) -> entry checker or None; called LIFTED
static PyObject *dynamic;         // dict: type -> predicate(obj, name): a name outside the table the kind admits
static PyObject *calls_unlifted;  // set of callables called WITHOUT lifting (pure probes such as _restricted.active)
static PyObject *costs;           // dict: unlifted callable -> cost checker, run before the call (range's length cap)
static PyObject *not_in_dialect;  // the exception type, registered by _restricted
static PyObject *mutators;        // dict: type -> frozenset of method names that write their receiver
static PyObject *fresh;           // set: lifted callables (or a method's function) whose result is a new value

static int is_owned(PyObject *obj);
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

static _Py_hashtable_t *owned;
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
                (void)_Py_hashtable_set(owned, op, op);     /* on failure the value is simply not owned */
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
    else if (ntracked && is_tracked(tp)) {
        (void)_Py_hashtable_steal(owned, op);
    }
    else if (is_generator_type(tp) && _Py_hashtable_len(lifted_gens)) {
        (void)_Py_hashtable_steal(lifted_gens, op);
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
        (void)_Py_hashtable_set(owned, result, result);
    }
}

/* An in-place operator on `v` that writes it (the type has the in-place slot): only on an owned value. */
int
_PyRestricted_CheckInPlace(PyObject *v)
{
    if (is_owned(v)) {
        return 0;
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

   The native form of the JS dialect's size caps. CPython's RAW allocator domain receives every block over 512 bytes
   (pymalloc hands those to PyMem_RawMalloc) and the arena allocator every arena of small objects, so wrapping those
   two counts the heap a run adds, net of what it frees, at almost no cost per small object. A request that would
   take the run past its cap FAILS before anything is allocated (so "a" * 10**11 is refused in microseconds instead of
   touching 100 GB), and trips the run, so the MemoryError the caller raises ends as NotInDialect however it is
   handled. The wrappers are installed once, as hooks that call the previous allocator (the documented way to hook
   after initialization), and count only on the running thread while a run is in progress.

   Not counted: memory a C library takes from malloc directly, which includes numpy's array data (PyDataMem). A numpy
   leaf's cost and size are its manifest's (tier 1 bounds over shape), not this cap's. */

static PyMemAllocatorEx raw_orig;
static PyObjectArenaAllocator arena_orig;
static int mem_installed;
static int mem_counting;
static unsigned long mem_thread;
static long long mem_live, mem_peak, mem_limit;
static int mem_tripped;
static _Py_hashtable_t *mem_blocks;         /* block -> size, for the blocks a run allocated through RAW */

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
raw_malloc(void *ctx, size_t n)
{
    if (!mem_mine()) {
        return raw_orig.malloc(raw_orig.ctx, n);
    }
    if (!mem_admit((long long)n)) {
        return NULL;
    }
    return mem_record(raw_orig.malloc(raw_orig.ctx, n), n);
}

static void *
raw_calloc(void *ctx, size_t nelem, size_t elsize)
{
    if (!mem_mine()) {
        return raw_orig.calloc(raw_orig.ctx, nelem, elsize);
    }
    if (elsize != 0 && nelem > (size_t)PY_SSIZE_T_MAX / elsize) {
        mem_tripped = 1;
        return NULL;
    }
    size_t n = nelem * elsize;
    if (!mem_admit((long long)n)) {
        return NULL;
    }
    return mem_record(raw_orig.calloc(raw_orig.ctx, nelem, elsize), n);
}

static void *
raw_realloc(void *ctx, void *ptr, size_t n)
{
    if (!mem_mine()) {
        return raw_orig.realloc(raw_orig.ctx, ptr, n);
    }
    long long old = mem_forget(ptr);          /* 0 for a block from before the run: then all of n counts */
    if (!mem_admit((long long)n - old)) {
        if (old) {
            (void)_Py_hashtable_set(mem_blocks, ptr, (void *)(uintptr_t)old);
        }
        return NULL;
    }
    void *p = raw_orig.realloc(raw_orig.ctx, ptr, n);
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
raw_free(void *ctx, void *ptr)
{
    if (mem_mine()) {
        mem_live -= mem_forget(ptr);
    }
    raw_orig.free(raw_orig.ctx, ptr);
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

static int
mem_install(void)
{
    if (mem_installed) {
        return 0;
    }
    PyMem_GetAllocator(PYMEM_DOMAIN_RAW, &raw_orig);
    PyObject_GetArenaAllocator(&arena_orig);
    _Py_hashtable_allocator_t a = {ht_malloc, ht_free};
    mem_blocks = _Py_hashtable_new_full(_Py_hashtable_hash_ptr, _Py_hashtable_compare_direct, NULL, NULL, &a);
    if (mem_blocks == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    PyMemAllocatorEx raw = {NULL, raw_malloc, raw_calloc, raw_realloc, raw_free};
    PyObjectArenaAllocator arena = {NULL, arena_alloc, arena_free};
    PyMem_SetAllocator(PYMEM_DOMAIN_RAW, &raw);
    PyObject_SetArenaAllocator(&arena);
    mem_installed = 1;
    return 0;
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
    mem_counting = 1;
    return 1;
}

/* Stop counting (the outermost run). Returns whether the cap was hit; *peak is the most the run held at once. */
int
_PyRestricted_MemoryEnd(long long *peak)
{
    mem_counting = 0;
    _Py_hashtable_clear(mem_blocks);
    *peak = mem_peak;
    return mem_tripped;
}

int
_PyRestricted_MemoryTripped(void)
{
    return mem_counting && mem_tripped;
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
