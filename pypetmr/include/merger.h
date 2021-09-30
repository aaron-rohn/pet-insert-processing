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

    uint64_t nsingles = 0;
    uint64_t total_size = 0;

    SinglesMerger(std::vector<std::string>);
    operator bool() const;
    void find_rst();
    bool finished() const;
    int first_not_empty() const;
    void reload();

    Single next_event();
};

#endif
