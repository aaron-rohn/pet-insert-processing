#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

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
static PyObject *petmr_load(PyObject*, PyObject*);
static PyObject *petmr_store(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles, METH_VARARGS, "read PET/MRI insert singles data"},
    {"coincidences", petmr_coincidences, METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},
    {"load", petmr_load, METH_VARARGS, "Load coincidence listmode data"},
    {"store", petmr_store, METH_VARARGS, "Store coincidence listmode data"},
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

/*
 * Helpers to load and store data
 */

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

        long nl = n;

        arr = PyArray_SimpleNew(1, &nl, NPY_UINT8);
        std::memcpy(PyArray_DATA(arr), blk.data(), nl*sizeof(uint8_t));
        PyList_SetItem(lst, 0, arr);

        arr = PyArray_SimpleNew(1, &nl, NPY_UINT64);
        std::memcpy(PyArray_DATA(arr), TT.data(), nl*sizeof(uint64_t));
        PyList_SetItem(lst, 1, arr);

        for (size_t i = 0; i < energies.size(); i++)
        {
            arr = PyArray_SimpleNew(1, &nl, NPY_UINT16);
            std::memcpy(PyArray_DATA(arr), energies[i].data(), nl*sizeof(uint16_t));
            PyList_SetItem(lst, 2 + i, arr);
        }

        return lst;
    }
};

/*
 * Load singles data from several files,
 * provided in a python list
 */

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

/*
 * Sort coincidences between one or more singles data files
 */

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

    Single ev;
    std::vector<CoincidenceData> coincidence_data;
    CoincidenceSorter trues (output_file_str);

    uint64_t processed_bytes = 0;

    PyThreadState *_save = PyEval_SaveThread();
    do
    {
        ev = merger.next_event();
        auto new_ev = trues.add_event(ev);

        if (new_ev.size() > 0)
        {
            if (!trues.file_open())
                coincidence_data.insert(coincidence_data.end(), new_ev.begin(), new_ev.end());
            else
                CoincidenceData::write(trues.output_file, new_ev);
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
    return CoincidenceData::to_py_data(coincidence_data);
}

static PyObject*
petmr_load(PyObject *self, PyObject *args)
{
    const char *fname;
    if (!PyArg_ParseTuple(args, "s", &fname))
        return NULL;

    auto cd = CoincidenceData::read(std::string(fname));
    PyObject *df = CoincidenceData::to_py_data(cd);
    if (PyErr_Occurred()) return NULL;

    return df;
}

static PyObject*
petmr_store(PyObject *self, PyObject *args)
{
    const char *fname;
    PyObject *df;
    if (!PyArg_ParseTuple(args, "sO", &fname, &df))
        return NULL;

    auto cd = CoincidenceData::from_py_data(df);
    if (PyErr_Occurred()) return NULL;

    std::ofstream f(fname);
    CoincidenceData::write(f, cd);

    Py_RETURN_NONE;
}
