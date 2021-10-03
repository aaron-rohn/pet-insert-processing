#ifndef MERGER_H
#define MERGER_H

#include <vector>
#include <string>
#include <deque>
#include "singles.h"

class SinglesMerger
{
    const int tt_incr = 100;
    int expected_tt = 0;

    std::vector<SinglesReader> singles;
    std::vector<std::deque<Single>> events;

    public:

    uint64_t total_size = 0, nsingles = 0;
    SinglesMerger(std::vector<std::string>);
    void find_rst();
    operator bool() const;
    bool finished() const;
    void reload();
    std::vector<std::deque<Single>>::iterator first_not_empty();
    Single next_event();
};

#endif
