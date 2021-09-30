#include "singles.h"
#include "merger.h"

SinglesMerger::SinglesMerger(std::vector<std::string> fnames):
    events(SinglesReader::nblocks)
{
    for (auto f : fnames)
    {
        singles.emplace_back(f.c_str());
        total_size += singles.back().file_size;
    }
}

void SinglesMerger::find_rst()
{
    for (SinglesReader &sgl : singles)
        sgl.find_rst();
    expected_tt = 0;
}

bool SinglesMerger::finished()
{
    for (SinglesReader &sgl : singles)
        if (sgl) return false;
    return true;
}

int SinglesMerger::first_not_empty() const
{
    for (size_t i = 0; i < events.size(); i++)
        if (!events[i].empty()) return i;
    return -1;
}

Single SinglesMerger::next_event()
{
    Single e;
    int earliest;

    while (true)
    {
        earliest = first_not_empty();

        if (earliest == -1 && !finished()) reload_curr_ev();
        else break;
    }

    if (earliest != -1)
    {
        for (size_t i = 0; i < events.size(); i++)
        {
            auto &ev = events[i];

            if (!ev.empty() && ev.front() < events[earliest].front())
                earliest = i;
        }

        e = events[earliest].front();
        events[earliest].pop_front();
    }

    return e;
}

void SinglesMerger::reload_curr_ev()
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
