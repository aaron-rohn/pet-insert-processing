#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "singles.h"
#include "coincidence.h"
#include "sinogram.h"

#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *petmr_singles(PyObject*, PyObject*);
static PyObject *petmr_coincidences(PyObject*, PyObject*);
static PyObject *petmr_load(PyObject*, PyObject*);
static PyObject *petmr_store(PyObject*, PyObject*);
static PyObject *petmr_sort_sinogram(PyObject*, PyObject*);
static PyObject *petmr_load_sinogram(PyObject*, PyObject*);
static PyObject *petmr_save_sinogram(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles, METH_VARARGS, "read PET/MRI insert singles data"},
    {"coincidences", petmr_coincidences, METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},
    {"load", petmr_load, METH_VARARGS, "Load coincidence listmode data"},
    {"store", petmr_store, METH_VARARGS, "Store coincidence listmode data"},
    {"sort_sinogram", petmr_sort_sinogram, METH_VARARGS, "Sort coincidence data into a sinogram"},
    {"load_sinogram", petmr_load_sinogram, METH_VARARGS, "Load a sinogram from disk"},
    {"save_sinogram", petmr_save_sinogram, METH_VARARGS, "Save a sinogram to disk"},
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

std::streampos fsize(const char *fname)
{
    return std::ifstream(fname, std::ios::ate | std::ios::binary).tellg();
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
    std::vector<std::string> cfile_list;
    int nfiles = PyList_Size(file_list);
    for (int i = 0; i < nfiles; i++)
    {
        PyObject *item = PyList_GetItem(file_list, i);
        cfile_list.emplace_back(py_to_str(item));
    }
    return cfile_list;
}

/*
 * Load singles data from a single file
 */

static PyObject *
petmr_singles(PyObject *self, PyObject *args)
{
    const char *fname;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "s|K", &fname, &max_events))
        return NULL;

    std::ifstream f (fname, std::ios::binary);
    if (!f.good()) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    uint64_t nevents_approx = fsize(fname) / Record::event_size;

    uint64_t nevents = 0;
    uint8_t data[Record::event_size];
    std::vector<TimeTag> last_tt (Record::nmodules);
    std::vector<Single> events;

    Py_BEGIN_ALLOW_THREADS

    events.reserve(nevents_approx);

    while (f.good())
    {
        if (max_events > 0 && nevents > max_events) break;

        f.read((char*)data, Record::event_size);
        Record::align(f, data);

        auto mod = Record::get_module(data);

        if (Record::is_single(data))
            events.emplace_back(data, last_tt[mod]);
        else
            last_tt[mod] = TimeTag(data);
    }

    Py_END_ALLOW_THREADS

    return Single::to_py_data(events);
}

/*
 * Sort coincidence data from multiple provided singles files
 */

static PyObject *
petmr_coincidences(PyObject *self, PyObject *args)
{
    PyObject *status_queue, *terminate, *py_file_list;
    uint64_t max_events = 100'000'000;
    char *output_file_str = NULL;
    if (!PyArg_ParseTuple(args, "OOO|Ks", &terminate,
                                          &status_queue,
                                          &py_file_list,
                                          &max_events,
                                          &output_file_str)) return NULL;

    // Parse the input files
    auto file_list = pylist_to_strings(py_file_list);
    if (PyErr_Occurred()) return NULL;

    // Create handle to output file
    std::ofstream output_file_handle (
            output_file_str, std::ios::out | std::ios::binary);

    if (output_file_str && !output_file_handle)
    {
        PyErr_SetString(PyExc_IOError, strerror(errno));
        return NULL;
    }

    // Declare all the variables at the beginning
    // Allows for saving the thread state early on

    size_t sorter_threads = 8;
    size_t n = file_list.size();
    std::atomic_bool stop = false;

    // Items passed to time-tag scanning threads
    std::vector<std::thread> tt_scan;
    std::vector<std::mutex> all_lock (n);
    std::vector<std::condition_variable_any> all_cv (n);
    std::vector<std::queue<std::streampos>> all_pos (n);

    // Items used for sorting coincidences within a file range
    uint64_t ncoin = 0;
    std::vector<CoincidenceData> cd;
    std::deque<std::future<sorted_values>> workers;
    std::vector<std::streampos> start_pos(n), end_pos(n);

    // Begin multithreading
    PyThreadState *_save = PyEval_SaveThread();

    // calculate total size of all singles files
    std::streampos total_size = 0;
    for (auto &fname : file_list)
    {
        total_size += fsize(fname.c_str());
    }

    // Create time-tag search threads - one per file
    for (size_t i = 0; i < n; i++)
    {
        tt_scan.emplace_back(find_tt_offset,
                             file_list[i],
                             std::ref(all_lock[i]),
                             std::ref(all_cv[i]),
                             std::ref(all_pos[i]),
                             std::ref(stop));
    }

    // The first value returned is the reset position
    for (size_t i = 0; i < n; i++)
    {
        std::unique_lock<std::mutex> lck(all_lock[i]);
        all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });
        start_pos[i] = all_pos[i].front();
        all_pos[i].pop();
    }

    // verify that each file as readable and contained a reset
    if (std::any_of(start_pos.begin(), start_pos.end(),
                [](std::streampos p){ return p == -1; }))
    {
        stop = true;
        for (auto &th : tt_scan) th.join();

        std::string errstr("Error scanning for reset: ");
        errstr += strerror(errno);

        std::cout << errstr << std::endl;

        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError, errstr.c_str());
        return NULL;
    }

    while (!stop || workers.size() > 0)
    {
        // Get next time-tag increment to read until, for each file
        for (size_t i = 0; !stop && i < n; i++)
        {
            std::unique_lock<std::mutex> lck(all_lock[i]);
            all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });
            end_pos[i] = all_pos[i].front();
            all_pos[i].pop();

            if (end_pos[i] == -1) stop = true;
        }

        // If no file has completed yet, create a new worker to sort singles
        if (!stop)
        {
            workers.push_back(std::async(std::launch::async,
                &sort_span, file_list, start_pos, end_pos, std::ref(stop)));

            start_pos = end_pos;
        }

        // If a file is completed or there are sufficient workers, collect results
        if (stop || workers.size() >= sorter_threads)
        {
            auto [pos, new_cd] = workers.front().get();
            workers.pop_front();

            // either save the data or copy it to a buffer
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

            if ((term == Py_True) || (max_events > 0 && ncoin > max_events))
            {
                stop = true;
            }

            _save = PyEval_SaveThread();
        }
    }

    stop = true;
    for (auto &scanner : tt_scan)
        scanner.join();

    PyEval_RestoreThread(_save);

    return CoincidenceData::to_py_data(cd);
}

/*
 * Load listmode coincidence data and return a dataframe-like object
 */

static PyObject*
petmr_load(PyObject *self, PyObject *args)
{
    const char *fname;
    uint64_t max_events = 100'000'000;
    if (!PyArg_ParseTuple(args, "s|K", &fname, &max_events))
        return NULL;

    auto cd = CoincidenceData::read(fname, max_events);
    return CoincidenceData::to_py_data(cd);
}

/*
 * Store a dataframe-like object as listmode coincidence data
 */

static PyObject*
petmr_store(PyObject *self, PyObject *args)
{
    const char *fname;
    PyObject *df;
    if (!PyArg_ParseTuple(args, "sO", &fname, &df))
        return NULL;

    auto cd = CoincidenceData::from_py_data(df);
    if (PyErr_Occurred()) return NULL;

    std::ofstream f(fname, std::ios::binary);
    CoincidenceData::write(f, cd);
    Py_RETURN_NONE;
}

static PyObject*
petmr_sort_sinogram(PyObject *self, PyObject *args)
{
    const char *fname, *lut_dir;
    int flip_flood_y_coord = 0;
    uint64_t events_per_thread = 1'000'000;
    PyObject *terminate, *status_queue, *data_queue;
    if (!PyArg_ParseTuple(args, "ssOOO|iK",
                &fname,
                &lut_dir,
                &terminate,
                &status_queue,
                &data_queue,
                &flip_flood_y_coord,
                &events_per_thread)) return NULL;

    PyThreadState *_save = PyEval_SaveThread();

    Michelogram m(flip_flood_y_coord);
    std::string cfg_file = Michelogram::find_cfg_file(lut_dir);

    try
    {
        // catch invalid filename or json contents
        m.load_luts(lut_dir);
        m.load_photopeaks(cfg_file);
    }
    catch (std::invalid_argument &e)
    {
        PyEval_RestoreThread(_save);
        auto err_str = std::string("Error loading configuration data (") + e.what() + ")";
        PyErr_SetString(PyExc_RuntimeError, err_str.c_str());
        return NULL;
    };

    std::streampos coincidence_file_size = fsize(fname);
    std::streampos incr = sizeof(CoincidenceData) * events_per_thread;
    bool stop = false;

    size_t nworkers = 8;
    std::streampos start = 0;
    std::deque<std::future<std::streampos>> workers;

    while (!stop || workers.size() > 0)
    {
        if (!stop)
        {
            auto end = start + incr;
            if (end >= coincidence_file_size)
            {
                end = coincidence_file_size;
                stop = true;
            }

            workers.push_back(std::async(std::launch::async,
                &Michelogram::sort_span, &m, fname, start, end));

            start = end;
        }

        if (stop || workers.size() >= nworkers)
        {
            auto pos = workers.front().get();
            workers.pop_front();

            double perc = (double)pos / coincidence_file_size * 100;

            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "d", perc);
            if (term == Py_True) stop = true;
            _save = PyEval_SaveThread();
        }
    }

    PyEval_RestoreThread(_save);

    PyObject *data = m.to_py_data();
    PyObject_CallMethod(data_queue, "put", "O", data);
    Py_RETURN_NONE;
}

static PyObject*
petmr_load_sinogram(PyObject* self, PyObject* args)
{
    const char *sinogram_file;
    if (!PyArg_ParseTuple(args, "s", &sinogram_file))
        return NULL;

    Michelogram m;
    m.read_from(sinogram_file);
    return m.to_py_data();
}

static PyObject*
petmr_save_sinogram(PyObject* self, PyObject* args)
{
    const char *sinogram_file;
    PyObject *arr;
    if (!PyArg_ParseTuple(args, "sO", &sinogram_file, &arr))
        return NULL;

    Michelogram m(arr);
    m.write_to(sinogram_file);
    Py_RETURN_NONE;
}
