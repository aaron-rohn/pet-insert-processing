
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "coincidence.h"
#include <cmath>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

Coincidence::Coincidence(Single ev1, Single ev2) {
    if (ev1.block < ev2.block) {
        a = ev1;
        b = ev2;
    } else {
        a = ev2;
        b = ev1;
    }
}

SingleData::SingleData(const Single &s)
{
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
    auto sda = SingleData(a), sdb = SingleData(b);
    blk(a.block, b.block);
    tdiff((int64_t)a.abs_time - b.abs_time);
    e_a1(sda.e1);
    e_a2(sda.e2);
    e_b1(sdb.e1);
    e_b2(sdb.e2);
    x_a(std::round(sda.x * scale));
    y_a(std::round(sda.y * scale));
    x_b(std::round(sdb.x * scale));
    y_b(std::round(sdb.y * scale));
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
    auto begin = f.tellp();
    f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData));
    std::cout << "Wrote " << f.tellp() - begin << " bytes to file" << std::endl;
}

PyObject *CoincidenceData::to_py_data(
        const std::vector<CoincidenceData> &cd
) {
    PyObject *acols[ncol], *bcols[ncol];
    npy_intp nr = cd.size(), nrows[] = {nr};
    for (size_t i = 0; i < ncol-1; i++)
    {
        acols[i] = PyArray_SimpleNew(1, nrows, NPY_UINT16);
        bcols[i] = PyArray_SimpleNew(1, nrows, NPY_UINT16);
    }
    acols[ncol-1] = PyArray_SimpleNew(1, nrows, NPY_INT16);
    bcols[ncol-1] = PyArray_SimpleNew(1, nrows, NPY_INT16);

    for (npy_int i = 0; i < nr; i++)
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
        Py_INCREF(acols[i]);
        Py_INCREF(bcols[i]);

        // block, e1, e2, x, y
        PyList_SetItem(a, i, acols[i]);
        PyList_SetItem(b, i, bcols[i]);
    }

    Py_INCREF(a);
    Py_INCREF(b);

    PyObject *out = PyTuple_New(2);
    PyTuple_SetItem(out, 0, a);
    PyTuple_SetItem(out, 1, b);
    Py_INCREF(out);

    return out;
}

std::vector<CoincidenceData>
CoincidenceData::from_py_data(PyObject *obj)
{
    std::vector<CoincidenceData> cd;
    if (!PyTuple_Check(obj) || PyTuple_Size(obj) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "Object is not a tuple");
        return cd;
    }

    PyObject *a_df = PyTuple_GetItem(obj, 0);
    PyObject *b_df = PyTuple_GetItem(obj, 1);

    if (!PyList_Check(a_df) || !PyList_Check(b_df))
    {
        PyErr_SetString(PyExc_ValueError, "Member of tuple is not a list");
        return cd;
    }

    size_t ncol_actual = PyList_Size(a_df);
    std::cout << "Columns: " << ncol_actual << std::endl;

    // Numpy array must be uint16 and contiguous
    PyObject *acols[ncol], *bcols[ncol];
    for (size_t i = 0; i < ncol; i++)
    {
        acols[i] = PyList_GetItem(a_df, i);
        bcols[i] = PyList_GetItem(b_df, i);
    }

    // List of arrays must have the shape of a dataframe
    // All columns must have equal length
    npy_intp *shape = PyArray_DIMS(acols[0]);
    int nrow = shape[0];
    std::cout << "Rows: " << nrow << std::endl;

    cd.resize(nrow);
    
    for (int i = 0; i < nrow; i++)
    {
        cd[i].blk(*((uint16_t*)PyArray_GETPTR1(acols[0], i)),
                  *((uint16_t*)PyArray_GETPTR1(bcols[0], i)));

        cd[i].e_a1(*((uint16_t*)PyArray_GETPTR1(acols[1], i)));
        cd[i].e_a2(*((uint16_t*)PyArray_GETPTR1(acols[2], i)));

        cd[i].e_b1(*((uint16_t*)PyArray_GETPTR1(bcols[1], i)));
        cd[i].e_b2(*((uint16_t*)PyArray_GETPTR1(bcols[2], i)));

        cd[i].x_a(*((uint16_t*)PyArray_GETPTR1(acols[3], i)));
        cd[i].y_a(*((uint16_t*)PyArray_GETPTR1(acols[4], i)));

        cd[i].x_b(*((uint16_t*)PyArray_GETPTR1(bcols[3], i)));
        cd[i].y_b(*((uint16_t*)PyArray_GETPTR1(bcols[4], i)));

        cd[i].tdiff(*(( int16_t*)PyArray_GETPTR1(acols[5], i)));
    }

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
