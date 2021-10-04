#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <deque>
#include "singles.h"
#include "merger.h"

struct Coincidence
{
    Single a, b;
    Coincidence(Single, Single);
};

class CoincidenceSorter
{
    static const size_t vals_per_ev = 10;
    static const size_t scale = 511;

    const uint64_t delay;
    const uint64_t width;
    std::deque<Single> window;
    std::ofstream output_file;

    public:
    uint64_t counts = 0;
    CoincidenceSorter (const char *fname = NULL, uint64_t delay = 0, uint64_t width = 10):
        delay(delay), width(width), output_file(fname, std::ios::out | std::ios::binary) {
            if (fname) std::cout << "Output file: " << fname << std::endl;
    };

    std::deque<Coincidence> add_event (Single);
    void write_events(const std::deque<Coincidence>&);
    inline bool file_open() const { return bool(output_file); }
};

#endif
