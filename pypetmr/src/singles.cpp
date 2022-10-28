
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "singles.h"

#include <iostream>

PyObject* Single::to_py_data(std::vector<Single> &events)
{
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

void Record::align(std::istream &f, uint8_t data[])
{
    while (f.good() && !is_header(data[0]))
    {
        size_t n = event_size;
        for (size_t i = 1; i < event_size; i++)
        {
            if (is_header(data[i]))
            {
                std::memmove(data, data + i, event_size - i);
                n = i;
                break;
            }
        }
        read(f, data + event_size - n, n);
    }
}

bool Record::go_to_tt(
        std::ifstream &f,
        uint64_t value,
        std::atomic_bool &stop
) {
    uint8_t data[event_size];
    uint64_t last_tt_value = 0;
    bool synced = false;

    while(f.good() && !stop)
    {
        read(f, data);
        align(f, data);

        if (!is_single(data))
        {
            TimeTag tt(data);

            synced = (tt.value == (last_tt_value+1));
            last_tt_value = tt.value;

            if ((value == 0 && tt.value == 0) ||
                (synced && value > 0 && tt.value >= value))
            {
                break;
            }
        }
    }

    return !stop;
}

/*
 * Search a singles file for time tags, and provide the
 * file offset to a calling thread
 */

void Record::find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::tuple<uint64_t,std::streampos>> &q,
        std::atomic_bool &stop
) {
    static const std::streamoff offset (Record::event_size);
    uint64_t tt = 0, incr = 1000;
    std::ifstream f(fname, std::ios::binary);

    while (go_to_tt(f, tt, stop))
    {
        auto pos = f.tellg() - offset;

        {
            std::lock_guard<std::mutex> lg(l);
            q.push(std::make_tuple(tt,pos));
        }

        cv.notify_all();
        tt += incr;
    }

    {
        std::lock_guard<std::mutex> lg(l);
        q.push(std::make_tuple(0,-1));
    }

    cv.notify_all();
}
