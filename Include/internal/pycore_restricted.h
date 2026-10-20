#ifndef Py_INTERNAL_RESTRICTED_H
#define Py_INTERNAL_RESTRICTED_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_interpframe.h"   // _PyInterpreterFrame, PyStackRef_AsPyObjectBorrow()
#include "pycore_tstate.h"

/* window.ml restricted mode: is the gate active for this thread right now?

   Restricted for the C extent of _restricted.run(). Inside an approved C leaf the flag is lifted, and a frame whose
   code is CO_RESTRICTED above the lift barrier (a callback the leaf made, reached from C or inline through host
   Python) makes it restricted again. No per-frame bookkeeping: the walk runs only while lifted and stops at the
   barrier. */
static inline int
_PyRestricted_Active(PyThreadState *tstate)
{
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    if (ts->restricted) {
        return 1;
    }
    if (!ts->restricted_lifted) {
        return 0;
    }
    for (_PyInterpreterFrame *f = tstate->current_frame;
         f != NULL && f != ts->restricted_barrier;
         f = f->previous)
    {
        if (f->owner == FRAME_OWNED_BY_INTERPRETER) {
            continue;
        }
        PyObject *exec = PyStackRef_AsPyObjectBorrow(f->f_executable);
        if (PyCode_Check(exec) && (((PyCodeObject *)exec)->co_flags & CO_RESTRICTED)) {
            return 1;
        }
    }
    return 0;
}

/* Is a run active on this thread (restricted, or lifted inside one)? A CO_RESTRICTED frame may be entered only then
   (PYTHON_DIALECT.md, #895): the cell's own code runs only inside a run, never ungated and never gated outside one.
   Checked at RESUME, so a call, a generator's or coroutine's resumption and exec() of its code all pass through it. */
static inline int
_PyRestricted_InRun(PyThreadState *tstate)
{
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    return ts->restricted || ts->restricted_lifted;
}
/* Refuse entering `frame`'s CO_RESTRICTED code outside a run: -1 with NotInDialect set, naming the function. */
extern int _PyRestricted_RefuseOutside(struct _PyInterpreterFrame *frame);

/* Lift the gate for the C extent of an approved call (save, then restore on every exit). A CO_RESTRICTED frame
   reached inside it is gated again by _PyRestricted_Active(). */
typedef struct {
    int restricted;
    int lifted;
    struct _PyInterpreterFrame *barrier;
} _PyRestrictedSaved;

static inline _PyRestrictedSaved
_PyRestricted_Lift(PyThreadState *tstate)
{
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    _PyRestrictedSaved saved = {ts->restricted, ts->restricted_lifted, ts->restricted_barrier};
    ts->restricted = 0;
    ts->restricted_lifted = 1;
    ts->restricted_barrier = tstate->current_frame;
    return saved;
}

static inline void
_PyRestricted_Restore(PyThreadState *tstate, _PyRestrictedSaved saved)
{
    _PyThreadStateImpl *ts = (_PyThreadStateImpl *)tstate;
    ts->restricted = saved.restricted;
    ts->restricted_lifted = saved.lifted;
    ts->restricted_barrier = saved.barrier;
}

/* The gate (Python/restricted.c). Each Check returns 0 to proceed or -1 with NotInDialect set; call only while
   _PyRestricted_Active(). */
extern int _PyRestricted_CheckGetAttr(PyObject *obj, PyObject *name);
extern int _PyRestricted_CheckSetAttr(PyObject *obj, PyObject *name);
extern int _PyRestricted_CheckItem(PyObject *obj, int write);
extern int _PyRestricted_CheckImport(PyObject *name);
extern int _PyRestricted_CheckCall(PyObject *callable);
extern int _PyRestricted_IsLiftedLeaf(PyObject *callable);

/* The interpreter pushes a Python function's frame inline, which never reaches the call gate: a function registered as
   a LIFTED leaf (pandas Python) must take the gated path instead, or it would run as tier 3 (gated, and refused on its
   own internals). */
static inline int
_PyRestricted_MustCallThroughGate(PyThreadState *tstate, PyObject *callable)
{
    return _PyRestricted_Active(tstate) && _PyRestricted_IsLiftedLeaf(callable);
}
extern PyObject *_PyRestricted_Vectorcall(PyThreadState *tstate, PyObject *callable, PyObject *const *args,
                                          size_t nargsf, PyObject *kwnames);
extern PyObject *_PyRestricted_Call(PyThreadState *tstate, PyObject *callable, PyObject *args, PyObject *kwargs);

/* The step budget and the depth, charged at every periodic check while _PY_EVAL_RESTRICTED_BIT is set (the extent of
   _restricted.run()). Returns 0, or -1 with NotInDialect set. */
extern int _PyRestricted_Tick(PyThreadState *tstate);

/* What crosses into a lifted extent other than through a leaf's checker: item keys and values under "[]^", and a
   lifted dunder's operands. And whether a kind's Python dunders run lifted ("__^"). */
extern int _PyRestricted_CheckCrossing(PyObject *const *objs, Py_ssize_t n);
extern int _PyRestricted_LiftsDunders(PyObject *self);
/* Was this generator created inside a lifted extent (library code)? Then resuming it runs lifted. */
extern int _PyRestricted_IsLiftedGen(PyObject *gen);

/* Ownership (slice 5): is `obj` the cell's own; refuse an in-place operator on a value that is not; get the owned set
   ready before a run. */
PyAPI_FUNC(int) _PyRestricted_IsOwned(PyObject *obj);
extern int _PyRestricted_CheckInPlace(PyObject *v);
PyAPI_FUNC(int) _PyRestricted_OwnedReady(void);
/* Test hook: the sizes of the owned set and of the lifted generators' table. */
PyAPI_FUNC(void) _PyRestricted_TableSizes(Py_ssize_t *n_owned, Py_ssize_t *n_lifted_gens);

/* Memory (slice 5): a byte cap per run, counted at the RAW and arena allocators. */
PyAPI_FUNC(int) _PyRestricted_MemoryBegin(long long limit);
PyAPI_FUNC(int) _PyRestricted_MemoryEnd(long long *peak);
/* Is a sequence repeat's size within the cap? -1 when it is not (the run is tripped). Objects/abstract.c. */
extern int _PyRestricted_RepeatAdmit(PyObject *seq, PyObject *n);

/* Rollback (slice 6): the undo log at the write barrier. */
PyAPI_FUNC(int) _PyRestricted_WriteBarrier(PyObject *obj);
PyAPI_FUNC(int) _PyRestricted_UndoBegin(void);
PyAPI_FUNC(int) _PyRestricted_UndoEnd(int refused);
PyAPI_FUNC(int) _PyRestricted_AllowSnapshot(PyObject *type, PyObject *snapshot, PyObject *restore);

/* Registration, host-only (Modules/_restrictedmodule.c). */
PyAPI_FUNC(int) _PyRestricted_AllowKind(PyObject *type, PyObject *names);
PyAPI_FUNC(int) _PyRestricted_AllowModule(PyObject *name, PyObject *names);
PyAPI_FUNC(int) _PyRestricted_AllowCall(PyObject *callable, int lift, PyObject *checker);
PyAPI_FUNC(int) _PyRestricted_AllowDynamic(PyObject *type, PyObject *predicate);
PyAPI_FUNC(int) _PyRestricted_AllowMutators(PyObject *type, PyObject *names);
PyAPI_FUNC(int) _PyRestricted_AllowFresh(PyObject *callable);
extern int _PyRestricted_ReadAttrLifted(PyObject *obj, PyObject *name, PyObject **result);
PyAPI_FUNC(void) _PyRestricted_SetException(PyObject *exc);
PyAPI_FUNC(void) _PyRestricted_Reset(void);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_RESTRICTED_H */
