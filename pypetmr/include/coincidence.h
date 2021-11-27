#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <deque>
#include <queue>
#include <future>
#include <algorithm>
#include "singles.h"

#include <Python.h>

struct SingleData {
    double x1, y1, x2, y2, x, y;
    uint16_t e1 = 0, e2 = 0;
    SingleData() {};
    SingleData(const Single&);
};

struct CoincidenceData {
    static const long ncol = 6;
    static const size_t vals_per_ev = 10;
    static const size_t scale = 511;

    uint16_t data[vals_per_ev] = {0};

    CoincidenceData() {};
    CoincidenceData(const Single&, const Single&);

    static std::vector<CoincidenceData> read(std::string);
    static void write(std::ofstream&, const std::vector<CoincidenceData>&);

    static PyObject *to_py_data(const std::vector<CoincidenceData>&);
    static std::vector<CoincidenceData> from_py_data(PyObject*);

    inline uint8_t blka()  const { return data[0] >> 8; }
    inline uint8_t blkb()  const { return data[0] & 0xFF; }
    inline int16_t tdiff() const { return *((int16_t*)(&data[1])); }
    inline uint16_t e_a1() const { return data[2]; }
    inline uint16_t e_a2() const { return data[3]; }
    inline uint16_t e_b1() const { return data[4]; }
    inline uint16_t e_b2() const { return data[5]; }
    inline uint16_t x_a()  const { return data[6]; }
    inline uint16_t y_a()  const { return data[7]; }
    inline uint16_t x_b()  const { return data[8]; }
    inline uint16_t y_b()  const { return data[9]; }

    inline void blk(uint16_t a, uint16_t b) { data[0] = (a << 8) | b; }
    inline void tdiff(int16_t t)   { data[1] = *((uint16_t*)(&t)); }
    inline void e_a1(uint16_t val) { data[2] = val; }
    inline void e_a2(uint16_t val) { data[3] = val; }
    inline void e_b1(uint16_t val) { data[4] = val; }
    inline void e_b2(uint16_t val) { data[5] = val; }
    inline void  x_a(uint16_t val) { data[6] = val; }
    inline void  y_a(uint16_t val) { data[7] = val; }
    inline void  x_b(uint16_t val) { data[8] = val; }
    inline void  y_b(uint16_t val) { data[9] = val; }
};

void find_tt_offset(
        std::string,
        std::mutex&,
        std::condition_variable_any&,
        std::queue<std::streampos>&
);

using sorted_values = std::tuple<std::vector<std::streampos>,
                                 std::vector<CoincidenceData>>;

sorted_values sort_span(
        std::vector<std::string>,
        std::vector<std::streampos>,
        std::vector<std::streampos>
);

#endif
