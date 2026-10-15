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

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_RESTRICTED_H */
