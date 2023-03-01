#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "singles.h"
#include "coincidence.h"
#include "sinogram.h"

#include <future>

static PyObject *petmr_singles(PyObject*, PyObject*);
static PyObject *petmr_coincidences(PyObject*, PyObject*);
static PyObject *petmr_sort_sinogram(PyObject*, PyObject*);
static PyObject *petmr_save_listmode(PyObject*, PyObject*);
static PyObject *petmr_load_sinogram(PyObject*, PyObject*);
static PyObject *petmr_save_sinogram(PyObject*, PyObject*);
static PyObject *petmr_rebin_sinogram(PyObject*, PyObject*);
static PyObject *petmr_validate_singles_file(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles, METH_VARARGS, "read PET/MRI insert singles data"},
    {"coincidences", petmr_coincidences, METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},
    {"sort_sinogram", petmr_sort_sinogram, METH_VARARGS, "Sort coincidence data into a sinogram"},
    {"save_listmode", petmr_save_listmode, METH_VARARGS, "Convert coincidence format into a simple listmode format"},
    {"load_sinogram", petmr_load_sinogram, METH_VARARGS, "Load a sinogram from disk"},
    {"save_sinogram", petmr_save_sinogram, METH_VARARGS, "Save a sinogram to disk"},
    {"rebin_sinogram", petmr_rebin_sinogram, METH_VARARGS, "Simple SSRB for a Michelogram"},
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
    PyObject *mod = PyModule_Create(&petmrmodule);
    PyModule_AddIntConstant(mod, "nmodules", Geometry::nmodules);
    PyModule_AddIntConstant(mod, "nblocks", Geometry::nblocks);
    PyModule_AddIntConstant(mod, "nblocks_axial", Geometry::nblocks_axial);
    PyModule_AddIntConstant(mod, "ncrystals", Geometry::ncrystals);
    PyModule_AddIntConstant(mod, "ncrystals_total", Geometry::ncrystals_total);
    PyModule_AddIntConstant(mod, "nring", Geometry::nring);
    PyModule_AddIntConstant(mod, "ncrystals_per_ring", Geometry::ncrystals_per_ring);
    PyModule_AddIntConstant(mod, "dim_theta_full", Geometry::dim_theta_full);
    PyModule_AddIntConstant(mod, "dim_theta_half", Geometry::dim_theta_half);
    PyModule_AddIntConstant(mod, "dim_r", Geometry::dim_r);
    PyModule_AddIntConstant(mod, "ndoi", Geometry::ndoi);
    return mod;
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

void* Single::to_py_data(std::vector<Single> &events)
{
    // Move this function here so that singles and coincidences
    // dont have any references to python libraries

    // 'block', 'eF', 'eR', 'x', 'y'
    const int ncol = 5;

    PyArrayObject *cols[ncol];
    npy_intp nrow = events.size();

    for (size_t i = 0; i < ncol; i++)
    {
        cols[i] = (PyArrayObject*)PyArray_SimpleNew(1, &nrow, NPY_UINT16);
    }

    for (npy_int i = 0; i < nrow; i++)
    {
        const Single &s = events[i];
        SingleData sd(s);

        *((uint16_t*)PyArray_GETPTR1(cols[0], i)) = s.blk;
        *((uint16_t*)PyArray_GETPTR1(cols[1], i)) = sd.eF;
        *((uint16_t*)PyArray_GETPTR1(cols[2], i)) = sd.eR;
        *((uint16_t*)PyArray_GETPTR1(cols[3], i)) = sd.x;
        *((uint16_t*)PyArray_GETPTR1(cols[4], i)) = sd.y;
    }

    PyObject *a = PyList_New(ncol);
    for (size_t i = 0; i < ncol; i++)
        PyList_SetItem(a, i, (PyObject*)cols[i]);

    return a;
}

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
    std::vector<TimeTag> last_tt (Geometry::nmodules);
    std::vector<Single> events;

    PyThreadState *_save = PyEval_SaveThread();

    events.reserve(nevents_approx);

    while (f.good() && !stop)
    {
        nevents++;

        if (max_events > 0 && nevents > max_events)
            break;

        Record::read(f, data);
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
    return (PyObject*)Single::to_py_data(events);
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

    size_t nworkers = 7;
    size_t n = file_list.size();
    std::atomic_bool stop = false;

    // Items passed to time-tag scanning threads
    std::vector<std::thread> tt_scan;
    std::vector<std::mutex> all_lock (n);
    std::vector<std::condition_variable_any> all_cv (n);
    std::vector<std::queue<std::tuple<uint64_t,std::streampos>>> all_pos (n);

    // Items used for sorting coincidences within a file range
    uint64_t ncoin = 0;
    std::deque<std::future<SortedValues>> workers;
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
        tt_scan.emplace_back(Record::find_tt_offset,
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
    while (!stop || workers.size() > 0)
    {
        // Get next time-tag increment to read until, for each file
        for (size_t i = 0; !stop && i < n; i++)
        {
            std::unique_lock<std::mutex> lck(all_lock[i]);
            all_cv[i].wait(lck, [&]{ return !all_pos[i].empty(); });

            std::tie(current_tt[i], end_pos[i]) = all_pos[i].front();
            all_pos[i].pop();

            if (end_pos[i] == -1) stop = true;
        }

        if (!stop)
        {
            workers.push_back(std::async(std::launch::async,
                &CoincidenceData::coincidence_sort_span, file_list, start_pos, end_pos));

            start_pos = end_pos;
        }

        if (stop || workers.size() >= nworkers)
        {
            auto [pos, coin] = workers.front().get();
            workers.pop_front();

            CoincidenceData::write(output_file_handle, coin);
            if (!output_file_handle.good())
            {
                std::cout << "Failed to write data to coincidence file" << std::endl;
                stop = true;
            }

            // calculate data to update the UI
            ncoin += coin.size();
            auto proc_size = std::accumulate(pos.begin(), pos.end(), std::streampos(0));
            double perc = ((double)proc_size) / total_size * 100.0;

            // update the UI and interact with python
            PyEval_RestoreThread(_save);
            PyObject *term = PyObject_CallMethod(terminate, "is_set", "");
            PyObject_CallMethod(status_queue, "put", "((dK))", perc, ncoin);
            stop = stop || (term == Py_True) || (max_events > 0 && ncoin > max_events);
            _save = PyEval_SaveThread();
        }
    }

    stop = true;
    for (auto &scanner : tt_scan) scanner.join();

    PyEval_RestoreThread(_save);
    Py_RETURN_NONE;
}

static PyObject*
petmr_sort_sinogram(PyObject *self, PyObject *args)
{
    const char *fname;
    int prompts = 1, delays = 0, max_doi = Geometry::ndoi;
    PyObject *terminate, *status_queue, *data_queue;

    if (!PyArg_ParseTuple(args, "sppiOOO",
                &fname, &prompts, &delays, &max_doi,
                &terminate, &status_queue, &data_queue)) return NULL;

    PyThreadState *_save = PyEval_SaveThread();

    Michelogram m(Geometry::dim_theta_full, max_doi);

    std::streampos coincidence_file_size = fsize(fname);
    uint64_t events_per_thread = 1'000'000;
    std::streampos incr = sizeof(ListmodeData) * events_per_thread;
    bool stop = false;

    size_t nworkers = 8;
    std::streampos start = 0;
    std::deque<std::future<std::streampos>> workers;

    // spawn workers that add events to the michelogram
    while (!stop || workers.size() > 0)
    {
        if (!stop)
        {
            // create a worker for a specific span of the file
            auto end = std::min(start + incr, coincidence_file_size);
            stop = (end == coincidence_file_size);

            workers.push_back(std::async(std::launch::async,
                &Michelogram::sort_span, &m,
                fname, start, end, prompts, delays));

            start = end;
        }

        if (stop || workers.size() >= nworkers)
        {
            auto pos = workers.front().get();
            workers.pop_front();

            double perc = (double)pos / coincidence_file_size * 100;

            // update the python ui
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
    const char *lmfname, *fname, *cfgdir;
    double energy_window = 0.2;
    PyArrayObject *scaling_array, *fpos_array, *ppeak_array, *doi_array;
    PyObject *terminate, *status_queue, *data_queue;
    if (!PyArg_ParseTuple(args, "sssdOOOOOOO",
                &lmfname, &fname, &cfgdir, &energy_window,
                &scaling_array, &fpos_array,
                &ppeak_array, &doi_array,
                &terminate, &status_queue, &data_queue)) return NULL;

    npy_intp *shp = PyArray_DIMS(fpos_array);
    std::vector<uint64_t> fpos(shp[0]);
    std::memcpy(fpos.data(), PyArray_DATA(fpos_array), fpos.size()*sizeof(uint64_t));

    PyThreadState *_save = PyEval_SaveThread();

    Michelogram m(
            Geometry::dim_theta_full,
            Geometry::ndoi,
            energy_window,
            scaling_array, ppeak_array, doi_array);

    if (!m.loaded())
    {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_RuntimeError,
                "Error loading configuration data");
        return NULL;
    }

    std::ofstream lf (lmfname, std::ios::binary);

    size_t nworkers = 8;
    bool stop = false;
    std::streampos coincidence_file_size = fsize(fname);
    std::streamoff incr = sizeof(CoincidenceData)*1e6;
    int sz = fpos.size(), idx = 0;

    std::vector<char> buf(1024*4);
    std::deque<std::future<FILE*>> workers;
    std::streampos start = 0, processed = 0;

    // spawn workers to convert coincidence data to listmode data
    while (!stop || workers.size() > 0)
    {
        if (!stop)
        {
            // find the correct scaling factor for this starting file position
            for (; idx < sz-1 && fpos[idx+1] < (uint64_t)start; idx++) ;

            std::streampos end = std::min(start + incr, coincidence_file_size);
            stop = (end == coincidence_file_size);

            workers.push_back(std::async(std::launch::async,
                            &Michelogram::encode_span, &m,
                            fname, start, end, idx));
            start = end;
        }

        if (stop || workers.size() >= nworkers)
        {
            processed += incr;
            FILE *f = workers.front().get();
            workers.pop_front();
            std::fseek(f, 0, std::ios::beg);

            // append the temporary file to the main listmode file
            size_t n;
            while ((n = std::fread(buf.data(), sizeof(char), buf.size(), f)) > 0)
            {
                lf.write(buf.data(), sizeof(char)*n);
            }

            // delete the temp file
            std::fclose(f);

            // update the python ui
            double perc = (double)processed / coincidence_file_size * 100;
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

    Michelogram m(Geometry::dim_theta_half);

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

static PyObject*
petmr_rebin_sinogram(PyObject* self, PyObject* args)
{
    PyObject *arr;
    if (!PyArg_ParseTuple(args, "O", &arr))
        return NULL;

    Michelogram m(arr);

    int n = m.begin()->s.size(), dim_theta = n / Geometry::dim_r;
    npy_intp dims[] = {Geometry::nring*2 - 1,
                       dim_theta, Geometry::dim_r};

    PyArrayObject *rebinned = (PyArrayObject*)PyArray_SimpleNew(
            3, dims, npy_type);

    PyArray_FILLWBYTE(rebinned, 0);

    for (auto b = m.begin(), e = m.end(); b != e; ++b)
    {
        stype *sino_rebin = (stype*)PyArray_GETPTR3(
                rebinned, b.h + b.v, 0, 0);

        for (int j = 0; j < n; j++)
            sino_rebin[j] += b->s[j];
    }

    return (PyObject*)rebinned;
}

static PyObject*
petmr_validate_singles_file(PyObject* self, PyObject* args)
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
