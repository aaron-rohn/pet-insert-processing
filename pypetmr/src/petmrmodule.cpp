#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <iostream>
#include <cinttypes>
#include <cstdbool>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <future>

#include "singles.h"
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
    std::vector<std::string> cfile_list;
    int nfiles = PyList_Size(file_list);
    for (int i = 0; i < nfiles; i++)
    {
        PyObject *item = PyList_GetItem(file_list, i);
        cfile_list.emplace_back(py_to_str(item));
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
    const char *fname;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "s|K", &fname, &max_events))
        return NULL;

    EventRecords records;

    std::ifstream f (fname, std::ios::in | std::ios::binary);
    if (!f.good()) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    uint64_t nevents = 0;
    uint8_t data[Single::event_size];
    std::vector<TimeTag> last_tt (Single::nmodules);

    Py_BEGIN_ALLOW_THREADS
    while (f.good())
    {
        if (max_events > 0 && nevents > max_events) break;

        f.read((char*)data, Single::event_size);
        Single::align(f, data);

        auto mod = Single::get_module(data);

        if (Single::is_single(data))
        {
            records.append(Single(data, last_tt[mod]));
        }
        else
        {
            last_tt[mod] = TimeTag(data);
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
    PyObject *status_queue, *terminate, *py_file_list, *py_output_file;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "OOOO|K", &terminate,
                                          &status_queue,
                                          &py_file_list,
                                          &py_output_file,
                                          &max_events)) return NULL;

    // Parse the output file name (if provided)
    char *output_file_str = NULL;
    if (py_output_file != Py_None)
    {
        output_file_str = py_to_str(py_output_file);
        if (PyErr_Occurred()) return NULL;
    }

    // Parse the input files
    auto file_list = pylist_to_strings(py_file_list);
    if (PyErr_Occurred()) return NULL;

    // calculate total size of all singles files
    std::streampos total_size = 0;
    for (auto &fname : file_list)
    {
        std::ifstream f (fname, std::ios::ate | std::ios::binary);
        total_size += f.tellg();
    }

    std::ofstream output_file_handle (
            output_file_str, std::ios::out | std::ios::binary);

    size_t sorter_threads = 8;
    size_t n = file_list.size();

    std::vector<std::mutex> all_lock (n);
    std::vector<std::condition_variable_any> all_cv (n);
    std::vector<std::queue<std::streampos>> all_pos (n);

    std::vector<std::thread> tt_scan;

    // Create time-tag search threads - one per file
    for (size_t i = 0; i < n; i++)
    {
        tt_scan.emplace_back(find_tt_offset,
                             file_list[i],
                             std::ref(all_lock[i]),
                             std::ref(all_cv[i]),
                             std::ref(all_pos[i]));
    }

    std::vector<std::streampos> start_pos(n), end_pos(n);

    // The first value returned is the reset position
    for (size_t i = 0; i < n; i++)
    {
        std::unique_lock<std::mutex> lck(all_lock[i]);
        all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });
        start_pos[i] = all_pos[i].front();
        all_pos[i].pop();
    }

    // Verify that each thread found a reset
    if (std::any_of(start_pos.begin(), start_pos.end(),
                [](std::streampos i){ return i == -1; }))
    {
        for (auto &th : tt_scan) th.join();
        std::cout << "Failed to find reset" << std::endl;
        PyErr_SetString(PyExc_RuntimeError, "One or more files did not contain a reset");
        return NULL;
    }

    uint64_t ncoin = 0;

    std::vector<CoincidenceData> cd;
    std::deque<std::future<sorted_values>> workers;

    PyThreadState *_save = PyEval_SaveThread();

    bool done = false;

    while (!done || workers.size() > 0)
    {
        // Get next time-tag increment to read until, for each file
        for (size_t i = 0; !done && i < n; i++)
        {
            std::unique_lock<std::mutex> lck(all_lock[i]);
            all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });

            end_pos[i] = all_pos[i].front();
            all_pos[i].pop();

            done = (end_pos[i] == -1);
        }

        // If no file has completed yet, create a new worker to sort singles
        if (!done)
        {
            workers.push_back(std::async(std::launch::async,
                            &sort_span, file_list, start_pos, end_pos));

            start_pos = end_pos;
        }

        // If a file is completed or there are sufficient workers, collect results
        if (done || workers.size() >= sorter_threads)
        {
            auto [pos, new_cd] = workers.front().get();
            workers.pop_front();

            if (output_file_handle)
                CoincidenceData::write(output_file_handle, new_cd);
            else
                cd.insert(cd.end(), new_cd.begin(), new_cd.end());

            // calculate data to update the UI
            ncoin += new_cd.size();
            auto proc_size = std::accumulate(pos.begin(), pos.end(), std::streampos(0));
            double perc = ((double)proc_size) / total_size * 100.0;

            // update the UI and interact with python
            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "((dK))", perc, ncoin);
            _save = PyEval_SaveThread();

            // determine if stop is requested
            if (term == Py_True)
                done = true;

            // determine if maximum events exceeded
            if (max_events > 0 && ncoin > max_events)
                done = true;
        }
    }

    PyEval_RestoreThread(_save);

    for (auto &scanner : tt_scan)
        scanner.join();

    return CoincidenceData::to_py_data(cd);
}

static PyObject*
petmr_load(PyObject *self, PyObject *args)
{
    const char *fname;
    if (!PyArg_ParseTuple(args, "s", &fname))
        return NULL;

    auto cd = CoincidenceData::read(std::string(fname));
    return CoincidenceData::to_py_data(cd);
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
