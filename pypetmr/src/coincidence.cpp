
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include "coincidence.h"
#include <iostream>
#include <filesystem>

SingleData::SingleData(const Single &s)
{
    eR = 0;
    eF = 0;

    for (int i = 0; i < 4; i++)
    {
        eR += s.energies[i];
        eF += s.energies[i + 4];
    }

    /*            System Front
     *
     * (x = 0, y = 1)     (x = 1, y = 1)
     *               #####
     *               #D A#
     *               #   #
     *               #C B#
     *               #####
     * (x = 0, y = 0)     (x = 1, y = 0)
     *
     *            System Rear
     *
     * View of one block from outside the system looking inwards
     */

    // Fractional values 0-1
    xF = (double)(s.energies[A_FRONT] + s.energies[B_FRONT]) / eF;
    yF = (double)(s.energies[A_FRONT] + s.energies[D_FRONT]) / eF;
    xR = (double)(s.energies[A_REAR] + s.energies[B_REAR]) / eR;
    yR = (double)(s.energies[A_REAR] + s.energies[D_REAR]) / eR;

    // Pixel values 0-511
    x = std::round(xR * scale);
    y = std::round((yF+yR)/2.0 * scale);
}

CoincidenceData::CoincidenceData(const Single &a, const Single &b)
{
    const auto &[ev1, ev2] = a.blk < b.blk ?
        std::tie(a, b) : std::tie(b, a);

    SingleData sd1(ev1), sd2(ev2);

    // record absolute time in incr. of 100 ms
    uint64_t t = std::min(ev1.abs_time, ev2.abs_time);
    t /= (TimeTag::clks_per_tt * 100);
    abstime(t);

    int16_t td = ev1.abs_time - ev2.abs_time;
    tdiff(td <= window_width, td % window_delay);

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

std::vector<CoincidenceData>
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

    std::vector<CoincidenceData> cd(count);
    f.read((char*)cd.data(), count * sizeof(CoincidenceData));

    return cd;
}

void CoincidenceData::write(
        std::ofstream &f,
        const std::vector<CoincidenceData> &cd
) {
    f.write((char*)cd.data(), cd.size()*sizeof(CoincidenceData));
}

/*
 * Search a singles file for time tags, and provide the
 * file offset to a calling thread
 */

void find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::tuple<uint64_t,std::streampos>> &q,
        std::atomic_bool &stop
) {
    uint64_t tt = 0, incr = 1000;
    const std::streamoff offset (Record::event_size);
    std::ifstream f (fname, std::ios::in | std::ios::binary);

    while (Record::go_to_tt(f, tt, stop))
    {
        auto pos = f.tellg() - offset;

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

std::vector<CoincidenceData> CoincidenceData::sort(
        std::vector<Single> &singles
) {
    static const size_t ww = window_width, wd = window_delay;
    std::vector<CoincidenceData> coin;

    // iterate the first event in the coincidence - all events
    for (auto a = singles.begin(), d = singles.begin(), e = singles.end(); a != e; ++a)
    {
        // iterate the second event in the window for prompts
        for (auto b = a + 1;
             (b != e) && (time_difference(*b, *a) < ww);
             ++b)
        {
            if (valid_module(a->mod, b->mod))
                coin.emplace_back(*a, *b);
        }

        // look backwards for delays
        for(uint64_t tdiff; (tdiff = time_difference(*a, *d)) >= wd; ++d)
        {
            if (tdiff < (wd + ww) && valid_module(d->mod, a->mod))
                coin.emplace_back(*d, *a);
        }
    }

    return coin;
}

sorted_values coincidence_sort_span(
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

    // Load all the singles from each file
    for (size_t i = 0; i < n; i++)
    {
        Reader rdr (fnames[i], start_pos[i], end_pos[i]);

        while (rdr)
        {
            Record::read(rdr, rdr.data);
            Record::align(rdr, rdr.data);

            auto mod = Record::get_module(rdr.data);

            if (Record::is_single(rdr.data))
                singles.emplace_back(rdr.data, last_tt[mod]);
            else
                last_tt[mod] = TimeTag(rdr.data);
        }
    }

    // time-sort the singles and find prompts and delays
    std::sort(singles.begin(), singles.end());
    auto coin = CoincidenceData::sort(singles);
    return std::make_tuple(end_pos, coin);
}
