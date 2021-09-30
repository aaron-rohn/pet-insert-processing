#define PY_SSIZE_T_CLEAN

#include <iostream>
#include <cinttypes>
#include <cstdbool>
#include <vector>

#include "singles.h"
#include "merger.h"
#include "coincidence.h"

#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *petmr_singles(PyObject*, PyObject*);
static PyObject *petmr_coincidences(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles, METH_VARARGS, "read PET/MRI insert singles data"},
    {"coincidences", petmr_coincidences, METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef petmrmodule = {
    PyModuleDef_HEAD_INIT, "petmr", NULL, -1, petmrMethods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_petmr(void)
{
    import_array();
    return PyModule_Create(&petmrmodule);
}

void single_append(Single &s,
        std::vector<std::vector<uint16_t>> &energies,
        std::vector<uint8_t> &blocks,
        std::vector<uint64_t> &times
) {
    for (size_t i = 0; i < energies.size(); i++)
        energies[i].push_back(s.energies[i]);
    blocks.push_back(s.block);
    times.push_back(s.time);
}

static PyObject *
petmr_singles(PyObject *self, PyObject *args)
{
    const char *fname;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "s|K", &fname, &max_events))
        return NULL;

    // Open file and determine length
    SinglesReader reader(fname);
    if (!reader) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    // vectors to store output data
    std::vector<std::vector<uint16_t>> energies (Single::nch, std::vector<uint16_t>{});
    std::vector<uint8_t> blk;
    std::vector<uint64_t> TT;

    // read file contents until EOF or max_events
    while (reader.read())
    {
        if (max_events && reader.nsingles > max_events) break;

        if (reader.is_single)
        {
            single_append(reader.single, energies, blk, TT);
        }
    }

    // return data as numpy array

    PyObject *lst = PyList_New(energies.size() + 2);
    PyObject *arr = NULL;

    long int n = reader.nsingles;

    if (!lst) goto cleanup;

    arr = PyArray_SimpleNew(1, &n, NPY_UINT8);
    if (!arr) goto cleanup;
    std::memcpy(PyArray_DATA(arr), blk.data(), n*sizeof(uint8_t));
    PyList_SetItem(lst, 0, arr);

    arr = PyArray_SimpleNew(1, &n, NPY_UINT64);
    if (!arr) goto cleanup;
    std::memcpy(PyArray_DATA(arr), TT.data(), n*sizeof(uint64_t));
    PyList_SetItem(lst, 1, arr);

    for (size_t i = 0; i < energies.size(); i++)
    {
        arr = PyArray_SimpleNew(1, &n, NPY_UINT16);
        if (!arr) goto cleanup;
        std::memcpy(PyArray_DATA(arr), energies[i].data(), n*sizeof(uint16_t));
        PyList_SetItem(lst, 2 + i, arr);
    }

    return lst;

cleanup:
    for (size_t i = 0; lst && (i < energies.size() + 1); i++)
        Py_XDECREF(PyList_GetItem(lst, i));
    Py_XDECREF(arr);
    Py_XDECREF(lst);
    PyErr_SetString(PyExc_Exception, "Failed to create numpy array");
    return NULL;
}

static PyObject *
petmr_coincidences(PyObject *self, PyObject *args)
{
    PyObject *file_list;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "O|K", &file_list, &max_events))
        return NULL;

    std::vector<std::string> cfile_list;
    int nfiles = PyList_Size(file_list);
    for (int i = 0; i < nfiles; i++)
    {
        PyObject *item = PyList_GetItem(file_list, i);
        if (!PyUnicode_Check(item))
        {
            PyErr_SetString(PyExc_ValueError, "All list items must be strings");
            return NULL;
        }
        PyObject *utf8 = PyUnicode_AsUTF8String(item);
        cfile_list.push_back(PyBytes_AsString(utf8));
    }

    SinglesMerger merger (cfile_list);
    if (!merger) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    merger.find_rst();

    CoincidenceSorter trues;
    Single ev;

    do
    {
        ev = merger.next_event();
        auto c = trues.add_event(ev);
    }
    while (ev.valid && (max_events && merger.nsingles < max_events));

    PyObject *lst = PyList_New(0);
    if (!lst) goto cleanup;
    return lst;

cleanup:
    Py_XDECREF(lst);
    PyErr_SetString(PyExc_Exception, "Failed to create numpy array");
    return NULL;
}
