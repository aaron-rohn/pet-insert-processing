#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <condition_variable>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <queue>
#include <span>

#include "singles.h"

struct CoincidenceData;
using Coincidences = std::vector<CoincidenceData>;
using SortedValues = std::tuple<std::vector<off_t>, Coincidences>;

struct __attribute__((packed)) CoincidenceData
{
    // data members -> 22 bytes
    uint8_t blkb;
    uint8_t blka;
    int8_t tdiff;
    uint8_t prompt;
    uint16_t e_aF, e_aR;
    uint16_t e_bF, e_bR;
    uint16_t x_a, y_a;
    uint16_t x_b, y_b;
    uint16_t abstime;

    // Columns in python data - block, eF, eR, x, y
    static const long ncol = 5;
    static const size_t vals_per_ev = 11;

    // Fine-time LSB equals 1.389 ns (1 / (90e6 * 8) s)
    // Time window of 10->14ns, 20->28ns
    static const int16_t width = 10;
    static const int16_t delay = 100;

    CoincidenceData() {}

    // block a should always have the lower block number
    CoincidenceData(const SingleData &a, const SingleData &b):
        blkb(b.block), blka(a.block),
        tdiff(a.abstime - b.abstime),
        prompt(std::abs(tdiff) < width),
        e_aF(a.eF), e_aR(a.eR), e_bF(b.eF), e_bR(b.eR),
        x_a(a.x), y_a(a.y), x_b(b.x), y_b(b.y),
        abstime(std::min(a.abstime,b.abstime) / (CLK_PER_TT * 100)) {}

    static void write(std::ofstream &f, const Coincidences &cd)
    { f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData)); }

    inline std::tuple<uint8_t,uint8_t> blk() const
    { return std::make_tuple(blka, blkb); }

    inline std::tuple<uint16_t,uint16_t> e_sum() const
    { return std::make_tuple(e_aF+e_aR, e_bF+e_bR); }

    inline std::tuple<double,double> doi(double scale = 4096.0) const
    { return std::make_tuple((double)e_aF / (e_aF + e_aR) * scale,
                             (double)e_bF / (e_bF + e_bR) * scale); }

    inline std::tuple<uint16_t,uint16_t,uint16_t,uint16_t> pos() const
    { return std::make_tuple(x_a,y_a,x_b,y_b); }

    static void find_tt_offset(
            std::string, std::mutex&, std::condition_variable_any&,
            std::queue<std::tuple<uint64_t,off_t>>&,
            std::atomic_bool&);

    static Coincidences sort(const std::vector<SingleData>&);

    static SortedValues coincidence_sort_span(
            std::vector<std::string>,
            std::vector<off_t>,
            std::vector<off_t>);
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

// This class allows for taking ownership of a C array
// returned from the singles library

template <typename T>
class cspan
{
    public:
    T *data = nullptr;
    size_t size = 0;

    cspan() {};
    cspan(T* data, size_t size): data(data), size(size) {}
    cspan(cspan &&other) { *this = std::move(other); }

    cspan& operator=(cspan &&other)
    {
        data = other.data;
        size = other.size;
        other.data = nullptr;
        return *this;
    }

    ~cspan() { if (data) std::free(data); }
    T *begin() const { return data; }
    T *end() const { return data + size; }
};

// Wrappers for C library functions for better memory management
cspan<SingleData> span_singles_to_tt(SinglesReader*, uint64_t, uint64_t*, const SinglesFloodType);
cspan<SingleData> span_read_singles(const char*, off_t, off_t*, uint64_t*, const SinglesFloodType);

#endif
