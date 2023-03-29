#include "coincidence.h"
#include <cstring>

CoincidenceData::CoincidenceData(const SingleData &a, const SingleData &b, bool isprompt)
{
    // Event a is earlier (lower abstime), b is later (greater abstime)
    // Event 1 has the lower block number, 2 has the higher

    const auto &[ev1, ev2] = a.block < b.block ?
        std::tie(a, b) : std::tie(b, a);

    // record absolute time in incr. of 100 ms
    uint64_t t = a.abstime;
    t /= CLK_PER_TT*100;
    abstime(t);

    int8_t dt = ev1.abstime - ev2.abstime;
    tdiff(isprompt, dt);

    blk(ev1.block, ev2.block);
    e_aF(ev1.eF);
    e_aR(ev1.eR);
    e_bF(ev2.eF);
    e_bR(ev2.eR);
    x_a(ev1.x);
    y_a(ev1.y);
    x_b(ev2.x);
    y_b(ev2.y);
}

void CoincidenceData::find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::tuple<uint64_t,std::streampos>> &q,
        std::atomic_bool &stop
) {
    uint64_t tt = 0, incr = 1000;
    FILE *f = fopen(fname.c_str(), "rb");

    off_t pos;
    while ((pos = go_to_tt(f, tt)) > 0)
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
    std::vector<std::tuple<struct SingleData*,uint64_t>> vals;
    uint64_t nev = 0, total = 0;

    for (size_t i = 0; i < fnames.size(); i++)
    {
        struct SingleData *sgl = read_singles(
                fnames[i].c_str(), (off_t)start_pos[i], (off_t)end_pos[i], &nev);
        total += nev;
        vals.push_back(std::make_tuple(sgl,nev));
    }

    uint64_t cur = 0;
    std::vector<SingleData> singles(total);
    for (auto [sgl, len] : vals)
    {
        std::memcpy(&singles[cur], sgl, len*sizeof(SingleData));
        cur += len;
        free(sgl);
    }

    // time-sort the singles to ascending order
    std::sort(singles.begin(), singles.end(),
            [](const SingleData &a, const SingleData &b)
            { return a.abstime < b.abstime; });

    // generate prompts and delays
    auto coin = CoincidenceData::sort(singles);
    return std::make_tuple(end_pos, coin);
}

Coincidences CoincidenceData::sort(
        const std::vector<SingleData> &singles
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
                coin.emplace_back(*a, *b, isprompt);
        }
    }

    return coin;
}

