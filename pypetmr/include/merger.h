#ifndef COINCIDENCE_H
#define COINCIDENCE_H

#include <vector>
#include <string>
#include "singles.h"

class SinglesMerger
{
    const int tt_incr = 100;
    int expected_tt = 0;

    std::vector<SinglesReader> singles;
    std::vector<std::deque<Single>> events;
    void reload_curr_ev();

    public:

    uint64_t total_size = 0;

    SinglesMerger(std::vector<std::string>);
    void find_rst();
    bool finished();
    int first_not_empty() const;

    Single next_event();
};

#endif
