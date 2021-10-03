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
    const uint64_t delay;
    const uint64_t width;
    std::deque<Single> window;

    public:
    uint64_t counts = 0;
    CoincidenceSorter (uint64_t delay = 0, uint64_t width = 10):
        delay(delay), width(width) {};
    std::deque<Coincidence> add_event (Single);
};

#endif
