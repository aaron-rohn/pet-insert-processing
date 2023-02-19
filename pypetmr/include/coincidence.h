#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <iostream>
#include <algorithm>

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#include "singles.h"

struct CoincidenceData;
using Coincidences = std::vector<CoincidenceData>;
using SortedValues = std::tuple<std::vector<std::streampos>, Coincidences>;

struct CoincidenceData
{
    // Columns in python data - block, eF, eR, x, y
    static const long ncol = 5;
    static const size_t vals_per_ev = 11;

    // Fine-time LSB equals 1.389 ns (1 / (90e6 * 8) s)
    // Time window of 10->14ns, 20->28ns
    static const int16_t width = 20;
    static const int16_t delay = 100;

    // only data member of the struct is an array of uint16's
    uint16_t data[vals_per_ev] = {0};

    CoincidenceData() {};
    CoincidenceData(const Single&, const Single&, bool = true);

    static void write(std::ofstream &f, const Coincidences &cd)
    { f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData)); }

    inline uint8_t blka()  const { return data[0] >> 8; }
    inline uint8_t blkb()  const { return data[0] & 0xFF; }
    inline bool prompt()   const { return data[1] >> 8; }
    inline int8_t tdiff()  const { return int8_t(data[1] & 0xFF) % delay; }

    inline uint16_t e_aF() const { return data[2]; }
    inline uint16_t e_aR() const { return data[3]; }
    inline uint16_t e_bF() const { return data[4]; }
    inline uint16_t e_bR() const { return data[5]; }
    inline uint16_t x_a()  const { return data[6]; }
    inline uint16_t y_a()  const { return data[7]; }
    inline uint16_t x_b()  const { return data[8]; }
    inline uint16_t y_b()  const { return data[9]; }
    inline uint16_t abstime() const { return data[10]; }

    inline void blk(uint16_t a, uint16_t b)
    { data[0] = (a << 8) | b; }

    inline void tdiff(bool isprompt, int8_t dt)
    { data[1] = (isprompt << 8) | (dt & 0xFF); }

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
    { return std::make_tuple(blka(), blkb()); }

    inline std::tuple<uint16_t,uint16_t> e_sum() const
    { return std::make_tuple(e_aF()+e_aR(), e_bF()+e_bR()); }

    inline std::tuple<double,double> doi(double scale = 4096.0) const
    { return std::make_tuple(
            (double)e_aF() / (e_aF() + e_aR()) * scale,
            (double)e_bF() / (e_bF() + e_bR()) * scale); }
    
    inline std::tuple<uint16_t,uint16_t,uint16_t,uint16_t> pos() const
    { return std::make_tuple(x_a(),y_a(),x_b(),y_b()); }

    static Coincidences sort(const std::vector<Single>&);
    static SortedValues coincidence_sort_span(
            std::vector<std::string>,
            std::vector<std::streampos>,
            std::vector<std::streampos>);
};

struct ListmodeData {
    static const unsigned int invalid = 0x7F;

    // Fields within each short go from LSB to MSB moving down
    // so e.g. ring_a -> short[0][0:6]

    // short 0
    unsigned int ring_a     : 7 = invalid;
    unsigned int crystal_a  : 9;

    // short 1
    unsigned int ring_b     : 7;
    unsigned int crystal_b  : 9;

    // short 2
    unsigned int energy_b   : 6;
    unsigned int energy_a   : 6;
    unsigned int doi_b      : 2;
    unsigned int doi_a      : 2;

    // short 3
    unsigned int abstime    : 10;
    signed int tdiff        : 5;
    unsigned int prompt     : 1;

    bool valid() const { return ring_a != invalid; }
};

#endif
