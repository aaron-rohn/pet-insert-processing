
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "coincidence.h"
#include <cmath>
#include <tuple>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

SingleData::SingleData(const Single &s)
{
    e1 = 0;
    e2 = 0;
    for (int i = 0; i < 4; i++)
    {
        e1 += s.energies[i];
        e2 += s.energies[i + 4];
    }

    x1 = (double)(s.energies[0] + s.energies[1]) / e1;
    y1 = (double)(s.energies[0] + s.energies[3]) / e1;
    x2 = (double)(s.energies[4] + s.energies[5]) / e2;
    y2 = (double)(s.energies[4] + s.energies[7]) / e2;
    x = x1;
    y = (y1 + y2) / 2.0;
}
    
CoincidenceData::CoincidenceData(const Single &a, const Single &b)
{
    const auto &[ev1, ev2] = a.block < b.block ?
        std::tie(a, b) : std::tie(b, a);

    SingleData sd1(ev1), sd2(ev2);
    blk(ev1.block, ev2.block);
    tdiff((int64_t)ev1.abs_time - ev2.abs_time);
    e_a1(sd1.e1);
    e_a2(sd1.e2);
    e_b1(sd2.e1);
    e_b2(sd2.e2);
    x_a(std::round(sd1.x * scale));
    y_a(std::round(sd1.y * scale));
    x_b(std::round(sd2.x * scale));
    y_b(std::round(sd2.y * scale));
}

std::vector<CoincidenceData>
CoincidenceData::read(std::string fname)
{
    std::streampos fsize;

    {
        std::ifstream f(fname, std::ios::ate | std::ios::binary);
        fsize = f.tellg();
    }

    std::ifstream f(fname, std::ios::binary);
    uint64_t count = fsize / sizeof(CoincidenceData);
    count = count > 100'000'000 ? 100'000'000 : count;

    std::cout << "Reading " << count << " entries from coincidence file: " <<
        fname << std::endl;

    std::vector<CoincidenceData> cd(count);
    f.read((char*)cd.data(), count * sizeof(CoincidenceData));

    std::cout << "Finished reading" << std::endl;

    return cd;
}

void CoincidenceData::write(
        std::ofstream &f,
        const std::vector<CoincidenceData> &cd
) {
    f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData));
}

PyObject *CoincidenceData::to_py_data(
        const std::vector<CoincidenceData> &cd
) {
    PyObject *acols[ncol], *bcols[ncol];
    npy_intp nrow = cd.size();

    std::cout << "Begin creating python objects: 2x" <<
        ncol << "x" << nrow << std::endl;

    for (size_t i = 0; i < ncol-1; i++)
    {
        acols[i] = PyArray_SimpleNew(1, &nrow, NPY_UINT16);
        bcols[i] = PyArray_SimpleNew(1, &nrow, NPY_UINT16);
    }
    acols[ncol-1] = PyArray_SimpleNew(1, &nrow, NPY_INT16);
    bcols[ncol-1] = PyArray_SimpleNew(1, &nrow, NPY_INT16);

    for (npy_int i = 0; i < nrow; i++)
    {
        *((uint16_t*)PyArray_GETPTR1(acols[0], i)) = cd[i].blka();
        *((uint16_t*)PyArray_GETPTR1(acols[1], i)) = cd[i].e_a1();
        *((uint16_t*)PyArray_GETPTR1(acols[2], i)) = cd[i].e_a2();
        *((uint16_t*)PyArray_GETPTR1(acols[3], i)) = cd[i].x_a();
        *((uint16_t*)PyArray_GETPTR1(acols[4], i)) = cd[i].y_a();
        *(( int16_t*)PyArray_GETPTR1(acols[5], i)) = cd[i].tdiff();

        *((uint16_t*)PyArray_GETPTR1(bcols[0], i)) = cd[i].blkb();
        *((uint16_t*)PyArray_GETPTR1(bcols[1], i)) = cd[i].e_b1();
        *((uint16_t*)PyArray_GETPTR1(bcols[2], i)) = cd[i].e_b2();
        *((uint16_t*)PyArray_GETPTR1(bcols[3], i)) = cd[i].x_b();
        *((uint16_t*)PyArray_GETPTR1(bcols[4], i)) = cd[i].y_b();
        *(( int16_t*)PyArray_GETPTR1(bcols[5], i)) = cd[i].tdiff();
    }

    PyObject *a = PyList_New(ncol), *b = PyList_New(ncol);
    for (size_t i = 0; i < ncol; i++)
    {
        // block, e1, e2, x, y, tdiff
        PyList_SetItem(a, i, acols[i]);
        PyList_SetItem(b, i, bcols[i]);
    }
    PyObject *out = PyTuple_New(2);
    PyTuple_SetItem(out, 0, a);
    PyTuple_SetItem(out, 1, b);

    std::cout << "Finished creating python objects" << std::endl;

    return out;
}

std::vector<CoincidenceData>
CoincidenceData::from_py_data(PyObject *obj)
{
    std::vector<CoincidenceData> cd;
    PyObject *a, *b;
    size_t ncola = 0, ncolb = 0;
    PyObject *acols[ncol] = {NULL}, *bcols[ncol] = {NULL};
    npy_intp *ashape, *bshape;
    npy_intp nrow = 0;

    // Input must be a tuple with 2 items
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != 2) goto error;

    a = PyTuple_GetItem(obj, 0);
    b = PyTuple_GetItem(obj, 1);

    // Each tuple member must be a list
    if (!PyList_Check(a) || !PyList_Check(b)) goto error;

    ncola = PyList_Size(a);
    ncolb = PyList_Size(b);

    // Each list must have 6 members
    if (ncola != ncol || ncolb != ncol) goto error;

    for (size_t i = 0; i < ncol; i++)
    {
        acols[i] = PyList_GetItem(a, i);
        bcols[i] = PyList_GetItem(b, i);

        // Each list item must be a numpy array with one dimension
        if (PyArray_NDIM(acols[i]) != 1 || PyArray_NDIM(bcols[i]) != 1) goto error;

        ashape = PyArray_DIMS(acols[i]);
        bshape = PyArray_DIMS(bcols[i]);
        if (i == 0) nrow = ashape[0];

        // Each array must have the same number of items
        if (ashape[0] != nrow || bshape[0] != nrow) goto error;

        // The first five items must be uint16, the sixth must be int16
        if (i < ncol - 1)
        {
            if (PyArray_TYPE(acols[i]) != NPY_UINT16 ||
                PyArray_TYPE(bcols[i]) != NPY_UINT16) goto error;
        }
        else
        {
            if (PyArray_TYPE(acols[i]) != NPY_INT16 ||
                PyArray_TYPE(bcols[i]) != NPY_INT16) goto error;
        }
    }

    std::cout << "Number of coincidences: " << nrow << std::endl;
    cd.resize(nrow);
    
    for (npy_intp i = 0; i < nrow; i++)
    {
        cd[i].blk(*(uint16_t*)PyArray_GETPTR1(acols[0],i),
                  *(uint16_t*)PyArray_GETPTR1(bcols[0],i));
        cd[i].e_a1(*(uint16_t*)PyArray_GETPTR1(acols[1],i));
        cd[i].e_a2(*(uint16_t*)PyArray_GETPTR1(acols[2],i));
        cd[i].e_b1(*(uint16_t*)PyArray_GETPTR1(bcols[1],i));
        cd[i].e_b2(*(uint16_t*)PyArray_GETPTR1(bcols[2],i));
        cd[i].x_a(*(uint16_t*)PyArray_GETPTR1(acols[3],i));
        cd[i].y_a(*(uint16_t*)PyArray_GETPTR1(acols[4],i));
        cd[i].x_b(*(uint16_t*)PyArray_GETPTR1(bcols[3],i));
        cd[i].y_b(*(uint16_t*)PyArray_GETPTR1(bcols[4],i));
        cd[i].tdiff(*(int16_t*)PyArray_GETPTR1(acols[5],i));
    }

    return cd;

error:

    PyErr_SetString(PyExc_ValueError, "Invalid format for coincidence data");
    return cd;
}

/*
 * Search a singles file for time tags, and provide the
 * file offset to a calling thread
 */

void find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::streampos> &q
) {
    uint64_t tt_target = 0, tt_incr = 1000;
    std::ifstream f (fname, std::ios::in | std::ios::binary);

    while (Single::go_to_tt(f, tt_target))
    {
        tt_target += tt_incr;

        {
            std::lock_guard<std::mutex> lg(l);
            q.push(f.tellg() - std::streamoff(Single::event_size));
        }
        cv.notify_all();
    }

    {
        std::lock_guard<std::mutex> lg(l);
        q.push(-1);
    }
    cv.notify_all();
}

/*
 * Sort coincidences between multiple singles data files
 * given a start and end position for each file
 */

sorted_values sort_span(
        std::vector<std::string> fnames,
        std::vector<std::streampos> start_pos,
        std::vector<std::streampos> end_pos
) {
    auto n = fnames.size();

    // Initialize input files
    std::vector<std::ifstream> files;
    for (auto &fn : fnames)
        files.emplace_back(fn, std::ios::in | std::ios::binary);

    // Calculate the approximate number of singles in the data stream
    std::streampos fsize_to_process = 0;
    for (size_t i = 0; i < n; i++)
        fsize_to_process += (end_pos[i] - start_pos[i]);
    size_t approx_singles = fsize_to_process / Single::event_size;

    // Allocate storage to load all the singles
    std::vector<Single> singles;
    singles.reserve(approx_singles);

    uint8_t data[Single::event_size];
    std::vector<TimeTag> last_tt (Single::nmodules);

    // Load all the singles from each file
    for (size_t i = 0; i < n; i++)
    {
        auto &f = files[i];
        f.seekg(start_pos[i]);

        while (f.tellg() < end_pos[i])
        {
            f.read((char*)data, Single::event_size);
            Single::align(f, data);

            auto mod = Single::get_module(data);

            if (Single::is_single(data))
            {
                // Create single and heap sort by absolute time
                singles.emplace_back(data, last_tt[mod]);
                std::push_heap(singles.begin(), singles.end());
            }
            else
            {
                // Update latest time tag for appropriate module
                last_tt[mod] = TimeTag(data);
            }
        }
    }

    // sort heap with ascending absolute time
    std::sort_heap(singles.begin(), singles.end());

    uint64_t width = 10;
    std::vector<CoincidenceData> coincidences;

    for (auto a = singles.begin(); a != singles.end(); ++a)
    {
        for (auto b = a + 1; b != singles.end() && (b->abs_time - a->abs_time < width); ++b)
        {
            if (a->mod != b->mod)
                coincidences.emplace_back(*a, *b);
        }
    }

    return std::tie(end_pos, coincidences);
}
