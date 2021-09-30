#ifndef SINGLES_H
#define SINGLES_H

#include <cinttypes>
#include <iostream>
#include <cstring>
#include <vector>
#include <deque>

#define BYTE_IS_HEADER(byte) ((byte >> 3) == 0x1F)
#define DATA_FLAG(data) (data[0] & 0x4)
#define DATA_BLK(data) (((data[0] << 4) | (data[1] >> 4)) & 0x3F)
#define DATA_MOD(data) (((data[0] << 2) | (data[1] >> 6)) & 0xF)
#define CLK_PER_TT 800000ULL

struct TimeTag {
    uint8_t mod;
    uint64_t value;

    TimeTag() : mod(0), value(0) {}
    TimeTag(uint8_t[]);
};

struct Single {
    static const int nch = 8;

    uint16_t energies[nch] = {0};
    uint8_t block;
    uint64_t time;
    uint64_t abs_time;

    Single(uint8_t[], TimeTag&);
    Single() {}
    bool operator<(const Single &rhs) { return abs_time < rhs.abs_time; }
};

class SinglesReader
{
    static const int event_size = 16;

    FILE *f;
    uint8_t data[event_size];
    uint64_t offset;
    std::vector<TimeTag> tt;
    bool tt_aligned = true;

    public:

    static const int nmodules = 16;
    static const int nblocks = 64;

    uint64_t nsingles = 0, ntimetag = 0, file_size = 0, file_elems = 0;
    uint8_t mod;

    bool is_single = false;
    Single single;
    TimeTag timetag;

    SinglesReader (const char*);
    ~SinglesReader () { fclose(f); }
    operator bool() const {return f && offset;}
    uint64_t find_rst();
    uint64_t read();
    std::vector<std::deque<Single>> go_to_tt(uint64_t);
};

#endif
