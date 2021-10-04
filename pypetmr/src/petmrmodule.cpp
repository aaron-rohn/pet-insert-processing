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

char *py_to_str(PyObject *obj)
{
    if (!PyUnicode_Check(obj))
    {
        PyErr_SetString(PyExc_ValueError, "Object is not a unicode string");
        return NULL;
    }
    PyObject *utf8 = PyUnicode_AsUTF8String(obj);
    return PyBytes_AsString(utf8);
}

std::vector<std::string>
pylist_to_strings(PyObject *file_list)
{
    PyErr_Clear();
    std::vector<std::string> cfile_list;
    int nfiles = PyList_Size(file_list);
    for (int i = 0; i < nfiles; i++)
    {
        PyObject *item = PyList_GetItem(file_list, i);
        cfile_list.push_back(py_to_str(item));
    }
    return cfile_list;
}

struct EventRecords
{
    // vectors to store output data
    std::vector<std::vector<uint16_t>> energies;
    std::vector<uint8_t> blk;
    std::vector<uint64_t> TT;
    uint64_t n = 0;

    EventRecords(): energies(Single::nch, std::vector<uint16_t>{}) {};

    void append(const Single &s)
    {
        n++;
        for (size_t i = 0; i < energies.size(); i++)
            energies[i].push_back(s.energies[i]);
        blk.push_back(s.block);
        TT.push_back(s.time);
    }

    PyObject *to_list()
    {
        PyObject *lst = PyList_New(energies.size() + 2);
        PyObject *arr = NULL;

        long int nl = n;

        if (!lst) goto cleanup;

        arr = PyArray_SimpleNew(1, &nl, NPY_UINT8);
        if (!arr) goto cleanup;
        std::memcpy(PyArray_DATA(arr), blk.data(), nl*sizeof(uint8_t));
        PyList_SetItem(lst, 0, arr);

        arr = PyArray_SimpleNew(1, &nl, NPY_UINT64);
        if (!arr) goto cleanup;
        std::memcpy(PyArray_DATA(arr), TT.data(), nl*sizeof(uint64_t));
        PyList_SetItem(lst, 1, arr);

        for (size_t i = 0; i < energies.size(); i++)
        {
            arr = PyArray_SimpleNew(1, &nl, NPY_UINT16);
            if (!arr) goto cleanup;
            std::memcpy(PyArray_DATA(arr), energies[i].data(), nl*sizeof(uint16_t));
            PyList_SetItem(lst, 2 + i, arr);
        }

        return lst;

cleanup:
        for (size_t i = 0; lst && (i < energies.size() + 1); i++)
            Py_XDECREF(PyList_GetItem(lst, i));
        Py_XDECREF(arr);
        Py_XDECREF(lst);
        return NULL;
    }
};

static PyObject *
petmr_singles(PyObject *self, PyObject *args)
{
    PyObject *file_list;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "O|K", &file_list, &max_events))
        return NULL;

    auto cfile_list = pylist_to_strings(file_list);
    if (PyErr_Occurred()) return NULL;

    EventRecords records;

    Py_BEGIN_ALLOW_THREADS
    for (auto fname : cfile_list)
    {
        // Open file and determine length
        SinglesReader reader(fname);
        if (!reader) 
        {
            PyErr_SetFromErrno(PyExc_IOError);
            return NULL;
        }

        // read file contents until EOF or max_events
        while (reader.read())
        {
            if (max_events && reader.nsingles > max_events) break;

            if (reader.is_single)
            {
                records.append(reader.single);
            }
        }
    }
    Py_END_ALLOW_THREADS

    PyObject *records_list = records.to_list();
    if (records_list == NULL)
    {
        PyErr_SetString(PyExc_Exception, "Failed to create numpy array");
        return NULL;
    }
    return records_list;
}

static PyObject *
petmr_coincidences(PyObject *self, PyObject *args)
{
    PyObject *status_queue, *terminate, *file_list, *output_file;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "OOOO|K", &terminate,
                                          &status_queue,
                                          &file_list,
                                          &output_file,
                                          &max_events)) return NULL;

    char *output_file_str = NULL;
    if (output_file != Py_None)
    {
        output_file_str = py_to_str(output_file);
        if (PyErr_Occurred()) return NULL;
    }

    auto cfile_list = pylist_to_strings(file_list);
    if (PyErr_Occurred()) return NULL;

    SinglesMerger merger (cfile_list);
    if (!merger)
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    merger.find_rst();

    EventRecords a, b;
    Single ev;
    CoincidenceSorter trues (output_file_str);

    uint64_t processed_bytes = 0;

    PyThreadState *_save = PyEval_SaveThread();
    do
    {
        ev = merger.next_event();
        auto c_all = trues.add_event(ev);
        if (!trues.file_open())
        {
            for (const auto &c : c_all)
            {
                a.append(c.a);
                b.append(c.b);
            }
        }
        else
        {
            trues.write_events(c_all);
        }

        processed_bytes += SinglesReader::event_size;
        double perc = ((double)processed_bytes / merger.total_size * 100);
        if (merger.nsingles % 10000 == 0)
        {
            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "((dK))", perc, trues.counts);
            _save = PyEval_SaveThread();
            if (term == Py_True) break;
        }
    }
    while (ev.valid && (max_events == 0 || trues.counts < max_events));
    PyEval_RestoreThread(_save);

    std::cout << "Number of coincidences: " << trues.counts << std::endl;

    PyObject *a_lst = a.to_list();
    PyObject *b_lst = b.to_list();
    PyObject *ab = PyTuple_New(2);
    if (!a_lst || !b_lst || !ab)
    {
        Py_XDECREF(a_lst);
        Py_XDECREF(b_lst);
        PyErr_SetString(PyExc_Exception, "Failed to create numpy array");
        return NULL;
    }

    PyTuple_SetItem(ab, 0, a_lst);
    PyTuple_SetItem(ab, 1, b_lst);
    return ab;
}
