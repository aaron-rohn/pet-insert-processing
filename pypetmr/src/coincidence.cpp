
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
    std::cout << "Reading " << count << " entries from coincidence file: " <<
        fname << std::endl;

    std::vector<CoincidenceData> cd(count);
    f.read((char*)cd.data(), fsize);

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

std::vector<CoincidenceData>
CoincidenceSorter::add_event(Single new_ev)
{
    std::vector<CoincidenceData> new_coins;
    auto ev = window.begin();
    for (; ev != window.end(); ++ev)
    {
        if (new_ev.abs_time - ev->abs_time > width) break;

        if (new_ev.mod != ev->mod)
            new_coins.emplace_back(new_ev,*ev);
    }

    counts += new_coins.size();
    window.erase(ev, window.end());
    new_ev.abs_time += delay;
    window.push_front(new_ev);
    return new_coins;
}
