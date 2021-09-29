#ifndef SINGLES_H
#define SINGLES_H

#include <cinttypes>
#include <iostream>
#include <cstring>

#define EV_SIZE 16
#define NMODULES 16
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
    uint16_t energies[8] = {0};
    uint8_t block;
    uint64_t time;
    uint64_t abs_time;

    Single(uint8_t[]);
    void set_abs_time(TimeTag&);
};

union Entry
{
    struct TimeTag timetag;
    struct Single single;
    Entry(): timetag(TimeTag()) {};
};

class SinglesReader
{
    FILE *f;
    uint8_t data[EV_SIZE];

    public:

    uint64_t nsingles = 0, ntimetag = 0;
    bool entry_is_single = false;
    Entry entry;
    uint8_t mod;

    SinglesReader (const char *fname) { f = fopen(fname, "rb"); }
    ~SinglesReader () { fclose(f); }
    bool operator!() { return !f; }
    void set_entry();
    uint64_t length();
    uint64_t find_rst();
    uint64_t read();
};

#endif
