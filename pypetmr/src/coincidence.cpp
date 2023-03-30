#include "coincidence.h"

void CoincidenceData::find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::tuple<uint64_t,std::streampos>> &q,
        std::atomic_bool &stop
) {
    uint64_t tt = 0, incr = 1000;
    SinglesReader rdr = reader_new_from_file(fname.c_str());

    off_t pos;
    while ((pos = go_to_tt(&rdr, tt)) > 0)
    {
        {
            std::lock_guard<std::mutex> lg(l);
            q.push(std::make_tuple(tt,pos));
        }

        cv.notify_all();
        tt += incr;
    }

    {
        std::lock_guard<std::mutex> lg(l);
        q.push(std::make_tuple(0,-1));
    }

    cv.notify_all();
}

/*
 * Sort coincidences between multiple singles data files
 * given a start and end position for each file
 */

SortedValues CoincidenceData::coincidence_sort_span(
        std::vector<std::string> fnames,
        std::vector<std::streampos> start_pos,
        std::vector<std::streampos> end_pos
) {
    std::vector<SingleData> singles;
    for (size_t i = 0; i < fnames.size(); i++)
    {
        uint64_t nev = 0;
        SingleData *p = read_singles(
                fnames[i].c_str(), start_pos[i], end_pos[i], &nev);

        auto s = std::span(p, nev);
        singles.insert(singles.end(), s.begin(), s.end());
        std::free(p);
    }

    std::ranges::sort(singles, {}, &SingleData::abstime);
    auto coin = CoincidenceData::sort(singles);
    return std::make_tuple(end_pos, coin);
}

Coincidences CoincidenceData::sort(
        const std::vector<SingleData>& singles
) {
    Coincidences coin;

    for (auto a = singles.begin(), e = singles.end(); a != e; ++a)
    {
        // a
        // |*** prompts ***| ----------- |*** delays ***| ---->
        // |<--- width --->|
        // |<------------ delay -------->|
        auto dend = a->abstime + delay + width;
        for (auto b = a + 1; b != e && b->abstime < dend; ++b)
        {
            auto dt = b->abstime - a->abstime;
            bool isprompt = dt < width, isdelay = dt >= delay;
            if ((isprompt || isdelay) && VALID_MODULE(MODULE(a->block), MODULE(b->block)))
            {
                const auto &[ev_a, ev_b] = a->block < b->block ?
                    std::tie(a, b) : std::tie(b, a);

                coin.emplace_back(*ev_a, *ev_b);
            }
        }
    }

    return coin;
}
