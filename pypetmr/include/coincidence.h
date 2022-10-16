#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <iostream>
#include <queue>
#include <future>
#include <algorithm>
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#include "singles.h"

struct CoincidenceData
{
    // Columns in python data - block, eF, eR, x, y
    static const long ncol = 5;
    static const size_t vals_per_ev = 11;

    // only data member of the struct is an array of uint16's
    uint16_t data[vals_per_ev] = {0};

    CoincidenceData() {};
    CoincidenceData(const Single&, const Single&);

    static std::vector<CoincidenceData> read(std::string, uint64_t=0);
    static void write(std::ofstream&, const std::vector<CoincidenceData>&);

    static PyObject *to_py_data(const std::vector<CoincidenceData>&);
    static std::vector<CoincidenceData> from_py_data(PyObject*);

    inline uint8_t blka()  const { return data[0] >> 8; }
    inline uint8_t blkb()  const { return data[0] & 0xFF; }
    inline int16_t tdiff() const { return *((int16_t*)&data[1]); }
    inline uint16_t e_aF() const { return data[2]; }
    inline uint16_t e_aR() const { return data[3]; }
    inline uint16_t e_bF() const { return data[4]; }
    inline uint16_t e_bR() const { return data[5]; }
    inline uint16_t x_a()  const { return data[6]; }
    inline uint16_t y_a()  const { return data[7]; }
    inline uint16_t x_b()  const { return data[8]; }
    inline uint16_t y_b()  const { return data[9]; }
    inline uint16_t abstime() const { return data[10]; }

    inline void blk(uint16_t a, uint16_t b) { data[0] = (a << 8) | b; }
    inline void tdiff(int16_t val) { data[1] = *((uint16_t*)&val); }
    inline void e_aF(uint16_t val) { data[2] = val; }
    inline void e_aR(uint16_t val) { data[3] = val; }
    inline void e_bF(uint16_t val) { data[4] = val; }
    inline void e_bR(uint16_t val) { data[5] = val; }
    inline void  x_a(uint16_t val) { data[6] = val; }
    inline void  y_a(uint16_t val) { data[7] = val; }
    inline void  x_b(uint16_t val) { data[8] = val; }
    inline void  y_b(uint16_t val) { data[9] = val; }
    inline void abstime(uint16_t val) { data[10] = val; }

    inline std::tuple<uint8_t,uint8_t> blk() const
    { return std::make_tuple(blka(), blkb()); };

    inline std::tuple<uint16_t,uint16_t> e_sum() const
    { return std::make_tuple(e_aF()+e_aR(), e_bF()+e_bR()); };

    inline std::tuple<double,double> doi(double scale = 4096.0) const
    { return std::make_tuple(
            (double)e_aF() / (e_aF() + e_aR()) * scale,
            (double)e_bF() / (e_bF() + e_bR()) * scale); }
    
    inline std::tuple<uint16_t,uint16_t,uint16_t,uint16_t> pos() const
    { return std::make_tuple(x_a(),y_a(),x_b(),y_b()); }
};

void find_tt_offset(
        std::string,
        std::mutex&,
        std::condition_variable_any&,
        std::queue<std::tuple<uint64_t,std::streampos>>&,
        std::atomic_bool&
);

using Coincidences = std::vector<CoincidenceData>;
using sorted_values = std::tuple<std::vector<std::streampos>, Coincidences, Coincidences>;

sorted_values sort_span(
        std::vector<std::string>,
        std::vector<std::streampos>,
        std::vector<std::streampos>,
        std::atomic_bool&);

#endif
