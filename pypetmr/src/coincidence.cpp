
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "coincidence.h"
#include <iostream>
#include <filesystem>

CoincidenceData::CoincidenceData(const Single &a, const Single &b, bool prompt)
{
    // Event a is earlier (lower abstime), b is later (greater abstime)

    const auto &[ev1, ev2] = a.blk < b.blk ?
        std::tie(a, b) : std::tie(b, a);

    SingleData sd1(ev1), sd2(ev2);

    // record absolute time in incr. of 100 ms
    uint64_t t = a.abs_time;
    t /= (TimeTag::clks_per_tt * 100);
    abstime(t);

    int64_t td = ev1.abs_time - ev2.abs_time;
    td %= window_width;

    tdiff(prompt, td);
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

Coincidences
CoincidenceData::read(std::string fname, uint64_t max_events)
{
    std::streampos fsize;

    {
        std::ifstream f(fname, std::ios::ate | std::ios::binary);
        fsize = f.tellg();
    }

    std::ifstream f(fname, std::ios::binary);
    uint64_t count = fsize / sizeof(CoincidenceData);
    count = max_events > 0 && count > max_events ? max_events : count;

    Coincidences cd(count);
    f.read((char*)cd.data(), count * sizeof(CoincidenceData));
    return cd;
}

void CoincidenceData::write(
        std::ofstream &f,
        const Coincidences &cd
) {
    f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData));
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

    // calculate the file size in bytes to read
    std::streampos fsize_to_process = 0;
    for (size_t i = 0; i < n; i++)
        fsize_to_process += (end_pos[i] - start_pos[i]);

    // estimate the number of singles to load
    const size_t approx_singles = fsize_to_process / Record::event_size;
    std::vector<Single> singles;
    singles.reserve(approx_singles);

    std::vector<TimeTag> last_tt (Record::nmodules);
    uint8_t data[Record::event_size];

    // Load all the singles from each file
    for (size_t i = 0; i < n; i++)
    {
        Reader rdr (fnames[i], start_pos[i], end_pos[i]);
        while (rdr)
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

    // time-sort the singles and find prompts and delays
    std::sort(singles.begin(), singles.end());
    auto coin = CoincidenceData::sort(singles);
    return std::make_tuple(end_pos, coin);
}

Coincidences CoincidenceData::sort(
        std::vector<Single> &singles
) {
    Coincidences coin;

    for (auto a = singles.begin(), d = singles.begin(), e = singles.end(); a != e; ++a)
    {
        // Identify prompt coincidences
        auto prompt_end = a->abs_time + window_width;
        for (auto b = a + 1; b != e && b->before(prompt_end); ++b)
        {
            if (a->valid_module(b->mod))
                coin.emplace_back(*a, *b, true);
        }

        // Identify delay coincidences
        uint64_t delay_end = a->abs_time - window_delay;
        uint64_t delay_start = delay_end - window_width;
        for (; d != a && d->before(delay_end); ++d)
        {
            if (d->after(delay_start) && d->valid_module(a->mod))
                coin.emplace_back(*d, *a, false);
        }
    }

    return coin;
}

