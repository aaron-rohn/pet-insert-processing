#include "coincidence.h"

CoincidenceData::CoincidenceData(const Single &a, const Single &b, bool isprompt)
{
    // Event a is earlier (lower abstime), b is later (greater abstime)
    // Event 1 has the lower block number, 2 has the higher

    const auto &[ev1, ev2] = a.blk < b.blk ?
        std::tie(a, b) : std::tie(b, a);

    SingleData sd1(ev1), sd2(ev2);

    // record absolute time in incr. of 100 ms
    uint64_t t = a.abs_time;
    t /= (TimeTag::clks_per_tt * 100);
    abstime(t);

    int8_t dt = ev1.abs_time - ev2.abs_time;
    tdiff(isprompt, dt);

    blk(ev1.blk, ev2.blk);
    e_aF(sd1.eF);
    e_aR(sd1.eR);
    e_bF(sd2.eF);
    e_bR(sd2.eR);
    x_a(sd1.x);
    y_a(sd1.y);
    x_b(sd2.x);
    y_b(sd2.y);
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
    const auto n = fnames.size();
    std::vector<std::streampos> fsizes;
    for (size_t i = 0; i < n; i++)
        fsizes.push_back(end_pos[i] - start_pos[i]);

    auto to_proc = std::accumulate(fsizes.begin(), fsizes.end(), 0);
    uint64_t approx_singles = to_proc / Record::event_size;

    std::vector<Single> singles;
    singles.reserve(approx_singles);

    std::vector<TimeTag> last_tt (Geometry::nmodules);
    uint8_t data[Record::event_size];

    // Load all the singles from each file
    for (size_t i = 0; i < n; i++)
    {
        Reader rdr (fnames[i], start_pos[i], end_pos[i]);
        while(rdr)
        {
            Record::read(rdr, data);
            Record::align(rdr, data);

            auto mod = Record::get_module(data);

            if (Record::is_single(data))
                singles.emplace_back(data, last_tt[mod]);
            else
                last_tt[mod] = TimeTag(data);
        }
    }

    // time-sort the singles to ascending order
    std::sort(singles.begin(), singles.end());

    // generate prompts and delays
    auto coin = CoincidenceData::sort(singles);
    return std::make_tuple(end_pos, coin);
}

Coincidences CoincidenceData::sort(
        const std::vector<Single> &singles
) {
    Coincidences coin;

    for (auto a = singles.begin(), e = singles.end(); a != e; ++a)
    {
        // a
        // |*** prompts ***| ----------- |*** delays ***| ---->
        // |<--- width --->|
        // |<------------ delay -------->|
        auto dend = a->abs_time + delay + width;
        for (auto b = a + 1; b != e && b->abs_time < dend; ++b)
        {
            auto dt = b->abs_time - a->abs_time;
            bool isprompt = dt < width, isdelay = dt >= delay;
            if ((isprompt || isdelay) && a->valid_module(b->mod))
                coin.emplace_back(*a, *b, isprompt);
        }
    }

    return coin;
}

