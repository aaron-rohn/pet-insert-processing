#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <queue>
#include <future>
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#include "singles.h"

struct CoincidenceData {
    // Columns in python data - block, e1, e2, x, y
    static const long ncol = 5;
    static const size_t vals_per_ev = 10;

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
    //inline uint16_t time() const { return data[1]; }
    inline  int16_t time() const { return *((int16_t*)&data[1]); }
    inline uint16_t e_a1() const { return data[2]; }
    inline uint16_t e_a2() const { return data[3]; }
    inline uint16_t e_b1() const { return data[4]; }
    inline uint16_t e_b2() const { return data[5]; }
    inline uint16_t x_a()  const { return data[6]; }
    inline uint16_t y_a()  const { return data[7]; }
    inline uint16_t x_b()  const { return data[8]; }
    inline uint16_t y_b()  const { return data[9]; }

    inline void blk(uint16_t a, uint16_t b) { data[0] = (a << 8) | b; }
    //inline void time(uint16_t val) { data[1] = val; }
    inline void time(int16_t val)  { data[1] = *((uint16_t*)&val); }
    inline void e_a1(uint16_t val) { data[2] = val; }
    inline void e_a2(uint16_t val) { data[3] = val; }
    inline void e_b1(uint16_t val) { data[4] = val; }
    inline void e_b2(uint16_t val) { data[5] = val; }
    inline void  x_a(uint16_t val) { data[6] = val; }
    inline void  y_a(uint16_t val) { data[7] = val; }
    inline void  x_b(uint16_t val) { data[8] = val; }
    inline void  y_b(uint16_t val) { data[9] = val; }

    inline std::tuple<uint8_t,uint8_t> blk() const
    { return std::make_tuple(blka(), blkb()); };

    inline std::tuple<uint16_t,uint16_t> e_sum() const
    { return std::make_tuple(e_a1()+e_a2(), e_b1()+e_b2()); };
    
    inline std::tuple<uint16_t,uint16_t,uint16_t,uint16_t> pos() const
    { return std::make_tuple(x_a(),y_a(),x_b(),y_b()); };
};

void find_tt_offset(
        std::string,
        std::mutex&,
        std::condition_variable_any&,
        std::queue<std::streampos>&,
        std::atomic_bool&
);

using sorted_values = std::tuple<std::vector<std::streampos>,
                                 std::vector<CoincidenceData>>;

sorted_values sort_span(
        std::vector<std::string>,
        std::vector<std::streampos>,
        std::vector<std::streampos>,
        std::atomic_bool&
);

#endif
