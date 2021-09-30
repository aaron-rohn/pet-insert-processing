#ifndef SINGLES_H
#define SINGLES_H

#include <cinttypes>
#include <iostream>
#include <cstring>
#include <vector>
#include <deque>

struct TimeTag {
    const static uint64_t clks_per_tt = 800'000;

    uint8_t mod;
    uint64_t value;

    TimeTag() : mod(0), value(0) {}
    TimeTag(uint8_t[]);
};

struct Single {
    static const int nch = 8;

    uint16_t energies[nch] = {0};
    uint8_t block;
    uint8_t mod;
    uint64_t time;
    uint64_t abs_time;

    bool valid = true;

    Single(uint8_t[], TimeTag&);
    Single(): valid(false) {}
    bool operator<(const Single &rhs) const { return abs_time < rhs.abs_time; }

    static inline bool is_header(uint8_t);
    static inline uint8_t get_flag(uint8_t[]);
    static inline uint8_t get_block(uint8_t[]);
    static inline uint8_t get_module(uint8_t[]);
};

class SinglesReader
{
    static const int event_size = 16;

    std::string fname;
    FILE *f;
    uint8_t data[event_size];
    uint64_t offset = 1;
    std::vector<TimeTag> tt;
    bool aligned = true;

    public:

    static const int nmodules = 16;
    static const int nblocks = 64;

    uint64_t nsingles = 0, ntimetag = 0, file_size = 0, file_elems = 0;
    uint8_t mod;

    bool is_single = false;
    Single single;
    TimeTag timetag;

    SinglesReader (std::string);
    ~SinglesReader () { if (f) fclose(f); }
    operator bool() const {return f && offset;}
    uint64_t find_rst();
    uint64_t read();
    std::vector<std::deque<Single>> go_to_tt(uint64_t);
};

#endif
