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
static PyObject *petmr_save_listmode(PyObject*, PyObject*);
static PyObject *petmr_load_sinogram(PyObject*, PyObject*);
static PyObject *petmr_save_sinogram(PyObject*, PyObject*);
static PyObject *petmr_validate_singles_file(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles, METH_VARARGS, "read PET/MRI insert singles data"},
    {"coincidences", petmr_coincidences, METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},
    {"load", petmr_load, METH_VARARGS, "Load coincidence listmode data"},
    {"store", petmr_store, METH_VARARGS, "Store coincidence listmode data"},
    {"sort_sinogram", petmr_sort_sinogram, METH_VARARGS, "Sort coincidence data into a sinogram"},
    {"save_listmode", petmr_save_listmode, METH_VARARGS, "Convert coincidence format into a simple listmode format"},
    {"load_sinogram", petmr_load_sinogram, METH_VARARGS, "Load a sinogram from disk"},
    {"save_sinogram", petmr_save_sinogram, METH_VARARGS, "Save a sinogram to disk"},
    {"validate_singles_file", petmr_validate_singles_file, METH_VARARGS, "Indicate if a singles file contains a reset and timetags for each module"},
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

std::tuple<std::vector<double>,std::vector<uint64_t>>
validate_scaling_array(PyArrayObject *scaling, PyArrayObject *fpos)
{
    auto nullarr = std::make_tuple(std::vector<double>(),std::vector<uint64_t>());

    if (PyArray_NDIM(scaling) != 1 || PyArray_NDIM(fpos) != 1)
    {
        PyErr_SetString(PyExc_RuntimeError, "Wrong number of dimensions for scaling array");
        return nullarr;
    }

    if (PyArray_TYPE(scaling) != NPY_DOUBLE || PyArray_TYPE(fpos) != NPY_ULONGLONG)
    {
        PyErr_SetString(PyExc_RuntimeError, "Wrong data type for scaling array");
        return nullarr;
    }

    npy_intp *scaling_shape, *fpos_shape;
    scaling_shape = PyArray_DIMS(scaling);
    fpos_shape = PyArray_DIMS(fpos);

    if (scaling_shape[0] != fpos_shape[0])
    {
        PyErr_SetString(PyExc_RuntimeError, "Size of scaling and time arrays are different");
        return nullarr;
    }

    size_t n = scaling_shape[0];
    std::vector<double> scaling_vec (n);
    std::memcpy(scaling_vec.data(), PyArray_DATA(scaling), n*sizeof(double));
    std::vector<uint64_t> fpos_vec (n);
    std::memcpy(fpos_vec.data(), PyArray_DATA(fpos), n*sizeof(uint64_t));

    return std::make_tuple(scaling_vec, fpos_vec);
}

/*
 * Load singles data from a single file
 */

static PyObject *
petmr_singles(PyObject *self, PyObject *args)
{
    PyObject *status_queue, *terminate;
    const char *fname;
    uint64_t max_events = 0;
    if (!PyArg_ParseTuple(args, "OOsK",
                          &terminate,
                          &status_queue,
                          &fname,
                          &max_events)) return NULL;

    std::ifstream f (fname, std::ios::binary);
    if (!f.good()) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    uint64_t nevents_approx = fsize(fname) / Record::event_size;
    if (max_events > 0) nevents_approx = std::min(nevents_approx, max_events);

    bool stop = false;
    uint64_t nevents = 0;
    uint8_t data[Record::event_size];
    std::vector<TimeTag> last_tt (Record::nmodules);
    std::vector<Single> events;

    PyThreadState *_save = PyEval_SaveThread();

    events.reserve(nevents_approx);

    while (f.good() && !stop)
    {
        nevents++;

        if (max_events > 0 && nevents > max_events)
            break;

        f.read((char*)data, Record::event_size);
        Record::align(f, data);

        auto mod = Record::get_module(data);

        if (Record::is_single(data))
            events.emplace_back(data, last_tt[mod]);
        else
            last_tt[mod] = TimeTag(data);

        if (nevents % 100'000 == 0)
        {
            double perc = ((double)nevents) / nevents_approx* 100.0;

            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "((dK))", perc, nevents);
            stop = (term == Py_True);
            _save = PyEval_SaveThread();
        }
    }

    PyEval_RestoreThread(_save);
    return Single::to_py_data(events);
}

/*
 * Sort coincidence data from multiple provided singles files
 */

static PyObject *
petmr_coincidences(PyObject *self, PyObject *args)
{
    PyObject *status_queue, *terminate, *py_file_list;
    uint64_t max_events = 0;
    char *output_file_str = NULL;
    if (!PyArg_ParseTuple(args, "OOOs|K",&terminate,
                                         &status_queue,
                                         &py_file_list,
                                         &output_file_str,
                                         &max_events)) return NULL;

    // Parse the input files
    auto file_list = pylist_to_strings(py_file_list);
    if (PyErr_Occurred()) return NULL;

    // Create handle to output file
    std::ofstream output_file_handle (
            output_file_str, std::ios::out | std::ios::binary);

    if (!output_file_handle)
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
    std::vector<std::queue<std::tuple<uint64_t,std::streampos>>> all_pos (n);

    // Items used for sorting coincidences within a file range
    uint64_t ncoin = 0;
    std::deque<std::future<sorted_values>> workers;
    std::vector<std::streampos> start_pos(n), end_pos(n);
    std::vector<uint64_t> current_tt(n);

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

        std::tie(current_tt[i], start_pos[i]) = all_pos[i].front();
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

    // spawn workers as the file is processed
    while (true)
    {
        // Get next time-tag increment to read until, for each file
        for (size_t i = 0; i < n; i++)
        {
            std::unique_lock<std::mutex> lck(all_lock[i]);
            all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });

            std::tie(current_tt[i], end_pos[i]) = all_pos[i].front();
            all_pos[i].pop();

            if (end_pos[i] == -1) goto exit;
        }

        // spawn a new worker each iteration
        workers.push_back(std::async(std::launch::async,
            &coincidence_sort_span, 
            file_list, start_pos, end_pos, std::ref(stop)));

        start_pos = end_pos;

        if (workers.size() >= sorter_threads)
        {
            auto [pos, coin] = workers.front().get();
            workers.pop_front();
            CoincidenceData::write(output_file_handle, coin);

            // calculate data to update the UI
            ncoin += coin.size();
            auto proc_size = std::accumulate(pos.begin(), pos.end(), std::streampos(0));
            double perc = ((double)proc_size) / total_size * 100.0;

            // update the UI and interact with python
            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "((dK))", perc, ncoin);
            stop = (term == Py_True);
            _save = PyEval_SaveThread();

            if (stop || (max_events > 0 && ncoin > max_events))
            {
                std::cout << "Stop coincidence sorting with " << ncoin << " events" << std::endl;
                goto exit;
            }
        }
    }

exit:

    stop = true;

    for (auto &scanner : tt_scan)
    {
        scanner.join();
    }

    while (workers.size() > 0)
    {
        workers.front().get();
        workers.pop_front();
    }

    PyEval_RestoreThread(_save);
    Py_RETURN_NONE;
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
    const char *fname;
    PyObject *terminate, *status_queue, *data_queue;
    int prompts = 1, delays = 0;
    if (!PyArg_ParseTuple(args, "sppOOO",
                &fname, &prompts, &delays,
                &terminate, &status_queue, &data_queue)) return NULL;

    PyThreadState *_save = PyEval_SaveThread();

    Michelogram m(Geometry::dim_theta_full,
            std::string(), vec<double>());

    std::streampos coincidence_file_size = fsize(fname);
    uint64_t events_per_thread = 1'000'000;
    std::streampos incr = sizeof(ListmodeData) * events_per_thread;
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
                &Michelogram::sort_span, &m,
                fname, start, end,
                prompts, delays));

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
petmr_save_listmode(PyObject* self, PyObject* args)
{
    const char *fname, *lmfname, *cfgdir;
    PyArrayObject *scaling_array, *fpos_array;
    PyObject *terminate, *status_queue, *data_queue;
    if (!PyArg_ParseTuple(args, "sssOOOOO",
                &lmfname, &fname, &cfgdir,
                &scaling_array, &fpos_array,
                &terminate, &status_queue, &data_queue)) return NULL;

    auto [scaling, fpos] =
        validate_scaling_array(scaling_array,fpos_array);

    if (scaling.size() == 0)
        return NULL;

    PyThreadState *_save = PyEval_SaveThread();

    Michelogram m(Geometry::dim_theta_full, cfgdir, scaling);

    if (!m.loaded())
    {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError,
                "Error loading configuration data");
        return NULL;
    }

    std::ifstream cf(fname, std::ios::binary);
    std::ofstream lf(lmfname, std::ios::binary);

    int n = 0, iters = 1e6;
    bool stop = false;
    std::streampos coincidence_file_size = fsize(fname);
    size_t sz = scaling.size(), idx = 0;
    uint64_t pos = 0;

    CoincidenceData c;
    while (!stop && cf)
    {
        pos = cf.tellg();
        cf.read((char*)&c, sizeof(c));

        for (; idx < sz-1 && fpos[idx+1] < pos; idx++) ;

        m.write_event(lf, c, idx);

        // update progress on UI
        n = (n + 1) % iters;
        if (n == 0)
        {
            double perc = (double)pos / coincidence_file_size * 100;
            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "d", perc);
            if (term == Py_True) stop = true;
            _save = PyEval_SaveThread();
        }
    }

    PyEval_RestoreThread(_save);
    PyObject_CallMethod(data_queue, "put", "O", Py_None);
    Py_RETURN_NONE;
}

static PyObject*
petmr_load_sinogram(PyObject* self, PyObject* args)
{
    const char *sinogram_file;
    if (!PyArg_ParseTuple(args, "s", &sinogram_file))
        return NULL;

    Michelogram m(Geometry::dim_theta_half, 
            std::string(), std::vector<double>());

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

static PyObject *petmr_validate_singles_file(PyObject* self, PyObject* args)
{
    const char *singles_file;
    if (!PyArg_ParseTuple(args, "s", &singles_file))
        return NULL;

    std::ifstream f (singles_file, std::ios::binary);
    if (!f.good()) 
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    const auto pred = [](bool v){ return v;};
    const auto all = [=](std::vector<bool> v){
        return std::all_of(v.begin(), v.end(), pred); };

    const int nmodules = 4;

    uint8_t data[Record::event_size];
    std::vector<bool> has_tt (nmodules, false);
    bool has_rst = false, valid = false;

    Py_BEGIN_ALLOW_THREADS

    while (f.good())
    {
        Record::read(f, data);
        Record::align(f, data);

        auto mod = Record::get_module(data) % nmodules;

        if (!Record::is_single(data))
        {
            has_rst = has_rst || (TimeTag(data).value == 0);
            has_tt[mod] = true;
            valid = has_rst && all(has_tt);

            if (valid) break;
        }
    }

    Py_END_ALLOW_THREADS

    if (valid) Py_RETURN_TRUE;
    else
    {
        return Py_BuildValue("(iiiii)",
                (int)has_rst, (int)has_tt[0], (int)has_tt[1],
                (int)has_tt[2], (int)has_tt[3]);
    }
}
