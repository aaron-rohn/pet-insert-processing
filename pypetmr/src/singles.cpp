
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "singles.h"

#include <iostream>

TimeTag::TimeTag(uint8_t data[])
{
    // Time tag
    // CRC | f |    b   |                     0's                    |             TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }
    //       0          1      2    3    4    5    6    7    8    9   10   11   12   13   14   15

    mod = Record::get_module(data);

    uint64_t upper = (data[10] << 16) | (data[11] << 8) | data[12];
    uint64_t lower = (data[13] << 16) | (data[14] << 8) | data[15];
    value = (upper << 24) | lower;
}

Single::Single(uint8_t data[], const TimeTag &tt)
{
    // Single event
    // CRC | f |    b   | D_REAR | C_REAR  | B_REAR | A_REAR  | D_FRONT |C_FRONT | B_FRONT |A_FRONT |       TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }
    //       0          1      2    3      4      5    6      7      8    9     10     11   12     13     14   15

    blk = Record::get_block(data);
    mod = Record::get_module(data);

    energies[D_REAR] = ((data[1] << 8) | data[2]) & 0xFFF;
    energies[C_REAR] = (data[3] << 4) | (data[4] >> 4);
    energies[B_REAR] = ((data[4] << 8) | data[5]) & 0xFFF;
    energies[A_REAR] = (data[6] << 4) | (data[7] >> 4);

    energies[D_FRONT] = ((data[7] << 8) | data[8]) & 0xFFF;
    energies[C_FRONT] = (data[9] << 4) | (data[10] >> 4);
    energies[B_FRONT] = ((data[10] << 8) | data[11]) & 0xFFF;
    energies[A_FRONT] = (data[12] << 4) | (data[13] >> 4);

    time = ((data[13] << 16) | (data[14] << 8) | data[15]) & 0xFFFFF;
    abs_time = tt.value * TimeTag::clks_per_tt + time;
}


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

SingleData::SingleData(const Single &s)
{
    eR = 0;
    eF = 0;

    for (int i = 0; i < 4; i++)
    {
        eR += s.energies[i];
        eF += s.energies[i + 4];
    }

    /*            System Front
     *
     * (x = 0, y = 1)     (x = 1, y = 1)
     *               #####
     *               #D A#
     *               #   #
     *               #C B#
     *               #####
     * (x = 0, y = 0)     (x = 1, y = 0)
     *
     *            System Rear
     *
     * View of one block from outside the system looking inwards
     */

    // Fractional values 0-1
    xF = (double)(s.energies[A_FRONT] + s.energies[B_FRONT]) / eF;
    yF = (double)(s.energies[A_FRONT] + s.energies[D_FRONT]) / eF;
    xR = (double)(s.energies[A_REAR] + s.energies[B_REAR]) / eR;
    yR = (double)(s.energies[A_REAR] + s.energies[D_REAR]) / eR;

    // Pixel values 0-511
    x = std::round(xR * scale);
    y = std::round((yF+yR)/2.0 * scale);
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
    uint64_t tt = 0, incr = 1000;
    const std::streamoff offset (Record::event_size);

    std::vector<char> buf(1024*1024*1024);
    std::ifstream f;
    f.rdbuf()->pubsetbuf(buf.data(), buf.size());
    f.open(fname, std::ios::in | std::ios::binary);

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
