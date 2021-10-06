#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <deque>
#include <utility>
#include "singles.h"
#include "merger.h"

#include <Python.h>

struct Coincidence
{
    Single a, b;
    Coincidence(Single, Single);
};

struct SingleData {
    double x1, y1, x2, y2, x, y;
    uint16_t e1, e2;
    SingleData(const Single&);
};

struct CoincidenceData {
    static const long ncol = 6;
    static const size_t vals_per_ev = 10;
    static const size_t scale = 511;

    uint16_t data[vals_per_ev] = {0};

    CoincidenceData() {};
    CoincidenceData(const Coincidence& c):
        CoincidenceData(c.a, c.b) {}
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

    inline void blk(uint8_t a, uint8_t b) { data[0] = a << 8 | b; }
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

class CoincidenceSorter
{
    const uint64_t delay;
    const uint64_t width;
    std::deque<Single> window;

    public:
    std::ofstream output_file;
    uint64_t counts = 0;
    CoincidenceSorter (const char *fname = NULL, uint64_t delay = 0, uint64_t width = 10):
        delay(delay), width(width), output_file(fname, std::ios::out | std::ios::binary) {
            if (fname) std::cout << "Output file: " << fname << std::endl;
    };

    std::vector<CoincidenceData> add_event (Single);
    inline bool file_open() const { return bool(output_file); }
};

#endif
