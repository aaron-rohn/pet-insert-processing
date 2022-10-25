#ifndef SINGLES_H
#define SINGLES_H

#include <cstring>
#include <vector>
#include <fstream>
#include <atomic>
#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#define A_FRONT 7
#define B_FRONT 6
#define C_FRONT 5
#define D_FRONT 4
#define A_REAR 3
#define B_REAR 2
#define C_REAR 1
#define D_REAR 0

namespace Record
{
    const int event_size = 16;
    const int nmodules = 16;

    inline int module_above(int mod)
    { return (mod + 1) % nmodules; };

    inline int module_below(int mod)
    { return (mod + nmodules - 1) % nmodules; };

    inline bool is_header(uint8_t b)
    { return (b >> 3) == 0x1F; };

    inline bool is_single(uint8_t d[])
    { return d[0] & 0x4; };

    inline uint8_t get_block(uint8_t d[])
    { return ((d[0] << 4) | (d[1] >> 4)) & 0x3F; };

    inline uint8_t get_module(uint8_t d[])
    { return ((d[0] << 2) | (d[1] >> 6)) & 0xF; };

    inline void read(std::istream &f, uint8_t d[], size_t n = event_size)
    { f.read((char*)d, n); };

    void align(std::istream&, uint8_t[]);
    bool go_to_tt(std::ifstream&, uint64_t, std::atomic_bool&);
};

struct TimeTag
{
    const static uint64_t clks_per_tt = 800'000;

    uint8_t mod;
    uint64_t value;
    TimeTag(uint8_t[]);
    TimeTag(): mod(0), value(0) {};
};

struct Single
{
    static const int nch = 8;
    static const int nblocks = 64;

    uint16_t energies[nch] = {0};
    uint8_t blk, mod;
    uint64_t time, abs_time;
    Single(uint8_t[], const TimeTag&);

    inline bool operator<(const Single &rhs) const
    { return abs_time < rhs.abs_time; }

    static PyObject* to_py_data(std::vector<Single>&);
};

struct SingleData
{
    constexpr static double scale = 511;

    double xF, yF, xR, yR;
    uint16_t eF = 0, eR = 0, x = 0, y = 0;
    SingleData() {};
    SingleData(const Single&);
};

// A class to load the singles data into memory and expose it as an istream
class Reader: std::streambuf, public std::istream
{
    std::ifstream base;
    std::vector<char> buf;

    public:
    uint8_t data[Record::event_size];

    Reader(std::string fname, std::streampos start, std::streampos end):
        std::istream(this),
        base(fname),
        buf(end - start)
    {
        base.seekg(start);
        base.read(buf.data(), buf.size());
        this->setg(buf.data(), buf.data(), buf.data() + buf.size());
    }
};

#endif
