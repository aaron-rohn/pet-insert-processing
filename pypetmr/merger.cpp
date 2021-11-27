#include "singles.h"
#include "merger.h"
#include <algorithm>

SinglesMerger::SinglesMerger(std::vector<std::string> fnames):
    events(SinglesReader::nmodules)
{
    for (auto f : fnames)
    {
        singles.emplace_back(f);
        total_size += singles.back().file_size;
    }
}

SinglesMerger::operator bool() const
{
    return std::all_of(singles.begin(), singles.end(),
            [](const SinglesReader &s){ return bool(s); });
}

bool SinglesMerger::finished() const
{
    return std::none_of(singles.begin(), singles.end(),
            [](const SinglesReader &s){ return bool(s); });
}

void SinglesMerger::find_rst()
{
    for (SinglesReader &sgl : singles) sgl.find_rst();
    expected_tt = 0;
}

std::vector<std::deque<Single>>::iterator
SinglesMerger::first_not_empty()
{
    return std::find_if_not(events.begin(), events.end(),
            [](std::deque<Single> d){ return d.empty(); });
}

void SinglesMerger::reload()
{
    expected_tt += tt_incr;
    for (SinglesReader &s : singles)
    {
        auto new_events = s.go_to_tt(expected_tt);
        auto e = events.begin(), n = new_events.begin();
        for (; e != events.end() && n != new_events.end(); ++e, ++n)
            e->insert(e->end(), n->begin(), n->end());
    }
}

Single SinglesMerger::next_event()
{
    auto fst = first_not_empty();
    while (fst == events.end())
    {
        if (finished()) return Single();
        reload();
        fst = first_not_empty();
    }

    for (auto nxt = fst + 1; nxt != events.end(); ++nxt)
    {
        if (!nxt->empty() && nxt->front() < fst->front())
            fst = nxt;
    }

    nsingles++;
    Single e = fst->front();
    fst->pop_front();
    return e;
}
