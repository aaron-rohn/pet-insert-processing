#ifndef SINGLES_H
#define SINGLES_H

#include <cstring>
#include <vector>
#include <fstream>
#include <atomic>
#include <queue>
#include <condition_variable>

#include <constants.h>

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

    inline int module_above(int mod)
    { return (mod + 1) % Geometry::nmodules; };

    inline int module_below(int mod)
    { return (mod + Geometry::nmodules - 1) % Geometry::nmodules; };

    // Single event
    // CRC | f |    b   | D_REAR | C_REAR  | B_REAR | A_REAR  | D_FRONT |C_FRONT | B_FRONT |A_FRONT |       TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }
    //       0          1      2    3      4      5    6      7      8    9     10     11   12     13     14   15

    inline bool is_header(const uint8_t b)
    { return (b >> 3) == 0x1F; };

    inline bool is_single(const uint8_t d[])
    { return d[0] & 0x4; };

    inline uint8_t get_block(const uint8_t d[])
    { return ((d[0] << 4) | (d[1] >> 4)) & 0x3F; };

    inline uint8_t get_module(const uint8_t d[])
    { return ((d[0] << 2) | (d[1] >> 6)) & 0xF; };

    inline uint16_t energy_d_rear(const uint8_t d[])
    { return (((d[1] << 8) | d[2]) & 0xFFF); }
        
    inline uint16_t energy_c_rear(const uint8_t d[])
    { return ((d[3] << 4) | (d[4] >> 4)); }

    inline uint16_t energy_b_rear(const uint8_t d[])
    { return (((d[4] << 8) | d[5]) & 0xFFF); }

    inline uint16_t energy_a_rear(const uint8_t d[])
    { return ((d[6] << 4) | (d[7] >> 4)); }

    inline uint16_t energy_d_front(const uint8_t d[])
    { return (((d[7] << 8) | d[8]) & 0xFFF); }

    inline uint16_t energy_c_front(const uint8_t d[])
    { return ((d[9] << 4) | (d[10] >> 4)); }

    inline uint16_t energy_b_front(const uint8_t d[])
    { return (((d[10] << 8) | d[11]) & 0xFFF); }

    inline uint16_t energy_a_front(const uint8_t d[])
    { return ((d[12] << 4) | (d[13] >> 4)); }

    inline uint64_t time(const uint8_t d[])
    { return ((d[13] << 16) | (d[14] << 8) | d[15]) & 0xFFFFF; }

    // Time tag
    // CRC | f |    b   |                     0's                    |             TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }
    //       0          1      2    3    4    5    6    7    8    9   10   11   12   13   14   15

    inline uint64_t timetag_upper(const uint8_t d[])
    { return (d[10] << 16) | (d[11] << 8) | d[12]; }

    inline uint64_t timetag_lower(const uint8_t d[])
    { return (d[13] << 16) | (d[14] << 8) | d[15]; }

    // other functions for generic singles listmode data

    inline void read(std::istream &f, uint8_t d[], size_t n = event_size)
    { f.read((char*)d, n); };

    void align(std::istream&, uint8_t[]);

    bool go_to_tt(std::ifstream&, uint64_t, std::atomic_bool&);

    void find_tt_offset(
            std::string, std::mutex&,
            std::condition_variable_any&,
            std::queue<std::tuple<uint64_t,std::streampos>>&,
            std::atomic_bool&
    );
};

struct TimeTag
{
    const static uint64_t clks_per_tt = 800'000UL;

    uint8_t mod;
    uint64_t value;

    TimeTag() {}
    TimeTag(uint8_t data[]):
        mod(Record::get_module(data)),
        value(Record::timetag_upper(data) << 24 |
              Record::timetag_lower(data)) {}
};

struct Single
{
    static const int nch = 8;

    uint8_t blk, mod;
    uint16_t energies[nch] = {0};
    uint64_t time, abs_time;

    Single() {};
    Single(uint8_t data[], const TimeTag &tt):
        blk(Record::get_block(data)),
        mod(Record::get_module(data)),
        energies {Record::energy_d_rear(data),
                  Record::energy_c_rear(data),
                  Record::energy_b_rear(data),
                  Record::energy_a_rear(data),
                  Record::energy_d_front(data),
                  Record::energy_c_front(data),
                  Record::energy_b_front(data),
                  Record::energy_a_front(data)},
        time(Record::time(data)),
        abs_time(tt.value * TimeTag::clks_per_tt + time) {}

    inline bool operator<(const Single &rhs) const
    { return abs_time < rhs.abs_time; }

    inline bool before(const uint64_t &time) const
    { return abs_time < time; }

    inline bool after(const uint64_t &time) const
    { return abs_time >= time; }

    inline bool valid_module(int m) const
    { return mod != m &&
             mod != Record::module_above(m) &&
             mod != Record::module_below(m); }

    static PyObject* to_py_data(std::vector<Single>&);
};

struct SingleData
{
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

    constexpr static double scale = 511;

    uint16_t eR, eF;
    double xF, yF, xR, yR;
    uint16_t x, y;

    SingleData() {};
    SingleData(const Single &s):
        eR(s.energies[D_REAR] + s.energies[C_REAR] +
           s.energies[B_REAR] + s.energies[A_REAR]),
        eF(s.energies[D_FRONT] + s.energies[C_FRONT] +
           s.energies[B_FRONT] + s.energies[A_FRONT]),

        // Fractional values 0-1
        xF((double)(s.energies[A_FRONT] + s.energies[B_FRONT]) / eF),
        yF((double)(s.energies[A_FRONT] + s.energies[D_FRONT]) / eF),
        xR((double)(s.energies[A_REAR] + s.energies[B_REAR]) / eR),
        yR((double)(s.energies[A_REAR] + s.energies[D_REAR]) / eR),

        // Pixel values 0-511
        x(std::round(xR * scale)),
        y(std::round((yF+yR)/2.0 * scale)) {}
};

// Object to load listmode data into memory and expose it as an istream

class Reader: std::streambuf, public std::istream
{
    std::vector<char> buf;

    public:

    Reader(std::string fname, std::streampos start, std::streampos end):
        std::istream(this), buf(end - start)
    {
        std::ifstream base(fname, std::ios::binary);
        base.seekg(start);
        base.read(buf.data(), buf.size());
        setg(buf.data(), buf.data(), buf.data() + buf.size());
    }
};

#endif
