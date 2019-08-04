// One formant modeling.

#include "Python.h"
#include "numpy/arrayobject.h"

enum cmp_op {
  cmp_smaller    = -2,
  cmp_smaller_eq = -1,
  cmp_equal      = 0,
  cmp_larger_eq  = 1,
  cmp_larger     = 2,
  cmp_not_equal  = 3,
};


template<class REAL>
inline
int find_1st_templ(REAL* x,  REAL limit, int stride, long size, cmp_op op) {

  REAL* pp = x;
  switch (op) {
  case  cmp_smaller:
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp < limit) 
        return ii;
    }
    break;
  case cmp_smaller_eq: 
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp <= limit) 
        return ii;
    }
    break;
  case cmp_equal :
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp == limit) 
        return ii;
    }
    break;
  case cmp_larger_eq :
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp >= limit) 
        return ii;
    }
    break;
  case cmp_larger :
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp > limit)  
        return ii;
    }
    break;
  case cmp_not_equal :
    for(long ii=0; ii <size; ++ii, pp += stride){
      if (*pp != limit) 
        return ii;
    }
    break;
  default:
    return -2;
    break;
  }
  return -1;
}

static PyObject *cc_find_1st(PyObject *self, PyObject *args);


struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef find_1st_methods[] = {
  {"find_1st", cc_find_1st, METH_VARARGS,"find_1st(arr, lim, op)\nreturns element to first index ind for that\n\n arr[i] OP(op) limit\n\n is True.\nOP(op) represents one of the comparison operators defined as follows:\n\n   OP(-2) -> < \n   OP(-1) -> <= \n   OP(0) ->  == \n   OP(1) -> >= \n   OP(2) -> >\n"},
  {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3

static int find_1st_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int find_1st_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "find_1st",
        NULL,
        sizeof(struct module_state),
        find_1st_methods,
        NULL,
        find_1st_traverse,
        find_1st_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_find_1st(void)
#else
#define INITERROR return

PyMODINIT_FUNC
initfind_1st(void)
#endif

{
   import_array();
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("find_1st", find_1st_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);
    char exception_text[16] = "find_1st.Error";
    st->error = PyErr_NewException(exception_text, NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *cc_find_1st(PyObject *dummy, PyObject *args) {
  int op = 0;
  double limit = 0;
  PyArrayObject *input;
  if (!PyArg_ParseTuple(args, "O!di:find_1st",
                        &PyArray_Type, &input,
                        &limit,
                        &op))  return NULL;
  if (NULL == input)  return NULL;
  if (PyArray_NDIM(input)>1) {
     PyErr_SetString(PyExc_ValueError,
         "cc_find_1st::Input array must be 1 dimensional.");
    return NULL;
  }
  
  int stride = PyArray_STRIDE(input, 0);
  int type   = PyArray_TYPE(input);
  int ret    = -1;
  switch(type) {
  case NPY_DOUBLE:
    ret = find_1st_templ(reinterpret_cast<double*>(PyArray_DATA(input)), static_cast<double>(limit),
                         stride/sizeof(double), static_cast<long>(PyArray_DIMS(input)[0]), cmp_op(op));
    break;
  case NPY_FLOAT:
    ret = find_1st_templ(reinterpret_cast<float*>(PyArray_DATA(input)), 
                         static_cast<float>(limit), stride/sizeof(float),
                         static_cast<long>(PyArray_DIMS(input)[0]), cmp_op(op));
    break;
  case NPY_INT64:
    ret = find_1st_templ(reinterpret_cast<long*>(PyArray_DATA(input)),
                         static_cast<long>(limit), stride/sizeof(long),
                         static_cast<long>(PyArray_DIMS(input)[0]), cmp_op(op));
    break;
  case NPY_INT32:
    ret = find_1st_templ(reinterpret_cast<int*>(PyArray_DATA(input)),
                         static_cast<int>(limit), stride/sizeof(int),
                         static_cast<long>(PyArray_DIMS(input)[0]), cmp_op(op));
    break;
  case NPY_BOOL:
    if((cmp_op(op) != cmp_equal) && (cmp_op(op) != cmp_not_equal)){
      PyErr_SetString(PyExc_ValueError,
                      "find_1st::Invalid cmparison operator for input data type bool, onnly cmp_equal and cmp_notequal are supported.");
      return NULL;
    }
    ret = find_1st_templ(reinterpret_cast<npy_bool*>(PyArray_DATA(input)),
                         static_cast<npy_bool>(limit), stride/sizeof(npy_bool),
                         static_cast<long>(PyArray_DIMS(input)[0]), cmp_op(op));
    break;
  default:
PyErr_SetString(PyExc_ValueError,
         "cc_find_1st::Input data type must be one of float64, float32, int64, int32, or bool.");
    return NULL;
  }

  if (ret == -2) {
    PyErr_SetString(PyExc_ValueError, "cc_find_1st::invalid operator.");
    return NULL;
}
  return Py_BuildValue("i", ret);
}
