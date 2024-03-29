#include <fstream>
#define _FILE_OFFSET_BITS 64
#define PY_SSIZE_T_CLEAN
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "singles.h"
#include "coincidence.h"
#include "sinogram.h"

#include <future>
#include <queue>
#include <deque>
#include <numeric>
#include <cmath>

static PyObject *petmr_singles(PyObject*, PyObject*);
static PyObject *petmr_coincidences(PyObject*, PyObject*);
static PyObject *petmr_save_listmode(PyObject*, PyObject*);
static PyObject *petmr_load_listmode(PyObject*, PyObject*);
static PyObject *petmr_save_sinogram(PyObject*, PyObject*);
static PyObject *petmr_load_sinogram(PyObject*, PyObject*);
static PyObject *petmr_rebin_sinogram(PyObject*, PyObject*);
static PyObject *petmr_listmode_to_arr(PyObject*, PyObject*);
static PyObject *petmr_validate_singles_file(PyObject*, PyObject*);

static PyMethodDef petmrMethods[] = {
    {"singles", petmr_singles,
        METH_VARARGS, "read PET/MRI insert singles data"},

    {"coincidences", petmr_coincidences,
        METH_VARARGS, "read PET/MRI insert singles data, and sort coincidences"},

    {"save_listmode", petmr_save_listmode,
        METH_VARARGS, "Convert coincidence data into packed listmode"},

    {"load_listmode", petmr_load_listmode,
        METH_VARARGS, "Sort packed listmode into a sinogram"},

    {"save_sinogram", petmr_save_sinogram,
        METH_VARARGS, "Save a sinogram to disk"},

    {"load_sinogram", petmr_load_sinogram,
        METH_VARARGS, "Load a sinogram from disk"},

    {"rebin_sinogram", petmr_rebin_sinogram,
        METH_VARARGS, "Simple SSRB for a Michelogram"},

    {"listmode_to_arr", petmr_listmode_to_arr,
        METH_VARARGS, "Load listmode data as numpy array"},

    {"validate_singles_file", petmr_validate_singles_file,
        METH_VARARGS, "Indicate if a singles file contains a reset and timetags for each module"},

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

bool term_is_set(PyObject *t)
{ return t && (PyObject_CallMethod(t, "is_set", "") == Py_True); }

template <typename... args>
PyObject *queue_put(PyObject *q, const char *fmt, args... objs)
{ return q ? PyObject_CallMethod(q, "put", fmt, objs...) : NULL; }

/*
 * Load singles data from a single file
 */

static PyObject *
petmr_singles(PyObject *self, PyObject *args)
{
    const char *fname;
    uint64_t max_events = 0;
    int flood_type = int(BOTH);
    PyObject *status_queue = NULL, *terminate = NULL;

    if (!PyArg_ParseTuple(args, "s|iKOO",
                &fname, &flood_type, &max_events,
                &terminate, &status_queue)) return NULL;

    off_t sz = fsize(fname), incr = sz / 50;
    off_t start = 0, end = incr;
    bool stop = false;
    std::vector<SingleData> svec;

    PyThreadState *_save = PyEval_SaveThread();

    while((start < sz) && (max_events == 0 || max_events > svec.size()) && !stop)
    {
        uint64_t n = max_events > 0 ? (max_events - svec.size()) : 0;

        cspan<SingleData> s = span_read_singles(fname, start, &end, &n, SinglesFloodType(flood_type));
        svec.insert(svec.end(), s.begin(), s.end());

        PyEval_RestoreThread(_save);
        stop = stop || term_is_set(terminate);
        queue_put(status_queue, "((dK))", (double)end / sz * 100, svec.size());
        _save = PyEval_SaveThread();

        start = end;
        end += incr;
    }

    PyEval_RestoreThread(_save);

    npy_intp dims[] = {(long)svec.size(), 5};
    PyArrayObject *singles = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);

    for (size_t i = 0; i < svec.size(); i++)
    {
        uint16_t *row = (uint16_t*)PyArray_GETPTR2(singles, i, 0);
        row[0] = svec[i].block;
        row[1] = svec[i].eF;
        row[2] = svec[i].eR;
        row[3] = svec[i].x;
        row[4] = svec[i].y;
    }

    return (PyObject*)singles;
}

/*
 * Sort coincidence data from multiple provided singles files
 */

static PyObject *
petmr_coincidences(PyObject *self, PyObject *args)
{
    PyObject *py_file_list;
    char *output_file_str;
    uint64_t max_events = 0;
    PyObject *status_queue = NULL, *terminate = NULL;

    if (!PyArg_ParseTuple(args, "Os|KOO",
                &py_file_list, &output_file_str,
                &max_events, &status_queue, &terminate)) return NULL;

    // Parse the input files
    auto file_list = pylist_to_strings(py_file_list);
    if (PyErr_Occurred()) return NULL;

    // Create handle to output file
    std::ofstream output_file_handle (
            output_file_str, std::ios::out | std::ios::binary);

    size_t nworkers = std::thread::hardware_concurrency() - 1;
    size_t n = file_list.size();
    std::atomic_bool stop = false;

    // Items passed to time-tag scanning threads
    std::vector<std::thread> tt_scan;
    std::vector<std::mutex> all_lock (n);
    std::vector<std::condition_variable_any> all_cv (n);
    std::vector<std::queue<std::tuple<uint64_t,off_t>>> all_pos (n);

    // Items used for sorting coincidences within a file range
    uint64_t ncoin = 0;
    std::deque<std::future<SortedValues>> workers;
    std::vector<off_t> start_pos(n), end_pos(n);
    std::vector<uint64_t> current_tt(n);

    // Begin multithreading
    PyThreadState *_save = PyEval_SaveThread();

    // calculate total size of all singles files
    off_t total_size = 0;
    for (auto &fname : file_list)
        total_size += fsize(fname.c_str());

    // Create time-tag search threads - one per file
    for (size_t i = 0; i < n; i++)
    {
        tt_scan.emplace_back(CoincidenceData::find_tt_offset,
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

    // verify that each file is readable and contained a reset
    if (std::any_of(start_pos.begin(), start_pos.end(),
                [](off_t p){ return p == -1; }))
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

            if (end_pos[i] < 1) stop = true;
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

            // calculate data to update the UI
            ncoin += coin.size();
            stop = stop || (max_events > 0 && ncoin > max_events);
            auto proc_size = std::accumulate(pos.begin(), pos.end(), off_t(0));
            double perc = ((double)proc_size) / total_size * 100.0;

            // update the UI and interact with python
            PyEval_RestoreThread(_save);
            stop = stop || term_is_set(terminate);
            queue_put(status_queue, "((dK))", perc, ncoin);
            _save = PyEval_SaveThread();
        }
    }

    stop = true;
    for (auto &scanner : tt_scan) scanner.join();

    PyEval_RestoreThread(_save);
    Py_RETURN_NONE;
}

static PyObject*
petmr_save_listmode(PyObject *self, PyObject *args)
{
    const char *lmfname, *fname;
    double energy_window;
    PyArrayObject *lut_array, *ppeak_array, *doi_array;

    PyObject *terminate = NULL, *status_queue = NULL;
    int64_t startpos = -1, stoppos = -1;

    /*
     * Arguments are:
     * 1. Listmode filename (output)
     * 2. Coincidence filename (input)
     * 3. Energy window
     * 4. Numpy array with LUT
     * 5. Numpy array with photopeaks (blk * xtal * doi)
     * 6. Numpy array with DOI (blk * xtal * doi)
     *
     * starting byte position in input file or -1
     * ending byte position in input file or -1
     * threading.Event to indicate termination
     * queue.Queue for progress bar
     */

    if (!PyArg_ParseTuple(args, "ssdOOO|LLOO",
                &lmfname, &fname, &energy_window,
                &lut_array, &ppeak_array, &doi_array,
                &startpos, &stoppos, &terminate, &status_queue)) return NULL;

    if (term_is_set(terminate)) Py_RETURN_NONE;

    auto flags = startpos > 0 ?
        std::ios::binary | std::ios::app :
        std::ios::binary | std::ios::trunc;
    std::ofstream lf (lmfname, flags);

    if (startpos < 0) startpos = 0;
    if (stoppos < 0) stoppos = fsize(fname);
    auto span = stoppos - startpos;

    if (stoppos < 0 || !lf.good())
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    PyThreadState *thr = PyEval_SaveThread();

    Michelogram m(Geometry::dim_theta_full, Geometry::ndoi, energy_window,
                  lut_array, ppeak_array, doi_array);

    int64_t incr = sizeof(CoincidenceData)*1e6, processed = 0;
    size_t nworkers = std::thread::hardware_concurrency() - 1;
    bool stop = false;

    std::vector<char> buf(1024*4);
    std::deque<std::future<FILE*>> workers;

    // spawn workers to convert coincidence data to listmode data
    while (!stop || workers.size() > 0)
    {
        if (!stop)
        {
            int64_t end = std::min(startpos + incr, stoppos);
            stop = (end == stoppos);

            workers.push_back(std::async(std::launch::async,
                            &Michelogram::save_listmode, &m, fname, startpos, end));
            startpos = end;
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
            double perc = (double)processed / span * 100;
            PyEval_RestoreThread(thr);
            queue_put(status_queue, "d", perc);
            stop |= term_is_set(terminate);
            thr = PyEval_SaveThread();
        }
    }

    PyEval_RestoreThread(thr);
    Py_RETURN_NONE;
}

static PyObject*
petmr_load_listmode(PyObject *self, PyObject *args)
{
    const char *fname;
    int prompts, delays, max_doi;
    PyObject *terminate = NULL, *status_queue = NULL;

    if (!PyArg_ParseTuple(args, "sppi|OO",
                &fname, &prompts, &delays, &max_doi,
                &terminate, &status_queue)) return NULL;

    std::streampos lm_file_size = fsize(fname);
    if (lm_file_size < 0)
    {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }

    PyThreadState *thr = PyEval_SaveThread();

    Michelogram m(Geometry::dim_theta_full, max_doi);
    uint64_t events_per_thread = 1'000'000;
    std::streampos incr = sizeof(ListmodeData) * events_per_thread;
    bool stop = false;

    size_t nworkers = std::thread::hardware_concurrency() - 1;
    std::streampos start = 0;
    std::deque<std::future<std::streampos>> workers;

    // spawn workers that add events to the michelogram
    while (!stop || workers.size() > 0)
    {
        if (!stop)
        {
            // create a worker for a specific span of the file
            auto end = std::min(start + incr, lm_file_size);
            stop = (end == lm_file_size);

            workers.push_back(std::async(std::launch::async,
                &Michelogram::add_to_sinogram, &m,
                fname, start, end, prompts, delays));

            start = end;
        }

        if (stop || workers.size() >= nworkers)
        {
            auto pos = workers.front().get();
            workers.pop_front();
            double perc = (double)pos / lm_file_size * 100;

            // update the python ui
            PyEval_RestoreThread(thr);
            queue_put(status_queue, "d", perc);
            stop |= term_is_set(terminate);
            thr = PyEval_SaveThread();
        }
    }

    PyEval_RestoreThread(thr);
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
petmr_listmode_to_arr(PyObject* self, PyObject* args)
{
    const char *fname;
    int64_t start = 0, end = -1;
    if (!PyArg_ParseTuple(args, "s|LL", &fname, &start, &end))
        return NULL;

    auto nev = fsize(fname) / sizeof(ListmodeData);
    end = end < 1 ? nev : end;
    start = start < 1 ? 0 : start;
    nev = end - start;

    std::vector<ListmodeData> ev(nev);

    // ra, xa, rb, xb, eb, ea, db, da, t, td, p
    npy_intp dims[] = {long(nev), 11};
    PyArrayObject *lm = (PyArrayObject*)PyArray_SimpleNew(
            2, dims, NPY_UINT16);

    std::ifstream f(fname, std::ios::binary);
    f.seekg(start * sizeof(ListmodeData));
    f.read((char*)ev.data(), nev * sizeof(ListmodeData));

    for (size_t i = 0; i < nev; i++)
    {
        uint16_t *row = (uint16_t*)PyArray_GETPTR2(lm, i, 0);
        row[0]  = ev[i].ring_a;
        row[1]  = ev[i].crystal_a;
        row[2]  = ev[i].ring_b;
        row[3]  = ev[i].crystal_b;
        row[4]  = ev[i].energy_b;
        row[5]  = ev[i].energy_a;
        row[6]  = ev[i].doi_b;
        row[7]  = ev[i].doi_a;
        row[8]  = ev[i].abstime;
        row[9]  = ev[i].tdiff;
        row[10] = ev[i].prompt;
    }

    return (PyObject*)lm;
}

static PyObject*
petmr_validate_singles_file(PyObject* self, PyObject* args)
{
    const char *singles_file;
    if (!PyArg_ParseTuple(args, "s", &singles_file))
        return NULL;

    uint8_t flags = 0;

    Py_BEGIN_ALLOW_THREADS
    flags = validate(singles_file);
    Py_END_ALLOW_THREADS

    if (flags == 0x1F) Py_RETURN_TRUE;
    else return Py_BuildValue("(iiiii)",
            (int)((flags >> 4) & 0x1),  // has rst
            (int)((flags >> 0) & 0x1),  // mod 3 has tt
            (int)((flags >> 1) & 0x1),  // mod 2 has tt
            (int)((flags >> 2) & 0x1),  // mod 1 has tt
            (int)((flags >> 3) & 0x1)); // mod 0 has tt
}
