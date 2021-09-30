#include "singles.h"
#include "merger.h"

SinglesMerger::SinglesMerger(std::vector<std::string> fnames):
    events(SinglesReader::nmodules)
{
    for (auto f : fnames)
    {
        singles.emplace_back(f);
        total_size += singles.back().file_size;
    }
}

SinglesMerger::operator bool() const {
    for (const SinglesReader &sgl : singles)
        if (!sgl) return false;
    return true;
}

bool SinglesMerger::finished() const
{
    for (const SinglesReader &sgl : singles)
        if (sgl) return false;
    return true;
}

void SinglesMerger::find_rst()
{
    for (SinglesReader &sgl : singles)
        sgl.find_rst();
    expected_tt = 0;
}

int SinglesMerger::first_not_empty() const
{
    for (size_t i = 0; i < events.size(); i++)
        if (!events[i].empty()) return i;
    return -1;
}

void SinglesMerger::reload()
{
    expected_tt += tt_incr;

    for (SinglesReader &s : singles)
    {
        auto new_ev = s.go_to_tt(expected_tt);
        for (size_t i = 0; i < events.size(); i++)
        {
            events[i].insert(events[i].end(), 
                             new_ev[i].begin(),
                             new_ev[i].end());
        }
    }
}

Single SinglesMerger::next_event()
{
    Single e;
    int earliest = first_not_empty();

    while (earliest == -1)
    {
        if (finished()) return e;
        reload();
        earliest = first_not_empty();
    }

    for (size_t i = earliest + 1; i < events.size(); i++)
    {
        auto &ev = events[i];

        if (!ev.empty() && ev.front() < events[earliest].front())
            earliest = i;
    }

    nsingles++;
    e = events[earliest].front();
    events[earliest].pop_front();
    return e;
}
