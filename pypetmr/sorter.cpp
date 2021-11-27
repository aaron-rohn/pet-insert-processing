#include <queue>
#include <algorithm>
#include <future>
#include "singles.h"
#include "coincidence.h"

void find_tt_offset(
        std::string fname,
        std::mutex &l,
        std::condition_variable_any &cv,
        std::queue<std::streampos> &q
) {
    uint64_t tt_target = 0, tt_incr = 1000;
    std::ifstream f (fname, std::ios::in | std::ios::binary);

    while (Single::go_to_tt(f, tt_target))
    {
        tt_target += tt_incr;

        {
            std::lock_guard<std::mutex> lg(l);
            q.push(f.tellg() - std::streamoff(Single::event_size));
        }
        cv.notify_all();
    }

    {
        std::lock_guard<std::mutex> lg(l);
        q.push(-1);
    }
    cv.notify_all();
}

std::vector<CoincidenceData>
sort_span(
        std::vector<std::string> fnames,
        std::vector<std::streampos> start_pos,
        std::vector<std::streampos> end_pos
) {
    auto n = fnames.size();

    // Initialize input files
    std::vector<std::ifstream> files;
    for (auto &fn : fnames)
        files.emplace_back(fn, std::ios::in | std::ios::binary);

    // Calculate the approximate number of singles in the data stream
    std::streampos fsize_to_process = 0;
    for (size_t i = 0; i < n; i++)
        fsize_to_process += (end_pos[i] - start_pos[i]);
    size_t approx_singles = fsize_to_process / Single::event_size;

    // Allocate storage to load all the singles
    std::vector<Single> singles;
    singles.reserve(approx_singles);

    uint8_t data[Single::event_size];
    std::vector<TimeTag> last_tt (Single::nmodules);

    // Load all the singles from each file
    for (size_t i = 0; i < n; i++)
    {
        auto &f = files[i];
        f.seekg(start_pos[i]);

        while (f.tellg() < end_pos[i])
        {
            f.read((char*)data, Single::event_size);
            Single::align(f, data);

            auto mod = Single::get_module(data);

            if (Single::is_single(data))
            {
                // Create single and heap sort by absolute time
                singles.emplace_back(data, last_tt[mod]);
                std::push_heap(singles.begin(), singles.end());
            }
            else
            {
                // Update latest time tag for appropriate module
                last_tt[mod] = TimeTag(data);
            }
        }
    }

    // sort heap with ascending absolute time
    std::sort_heap(singles.begin(), singles.end());

    uint64_t width = 10, delay = 0;
    std::deque<Single> window;
    std::vector<CoincidenceData> coincidences;

    for (auto &e : singles)
    {
        // Find coincidences matching the latest single

        auto ev = window.begin();
        for (; ev != window.end(); ++ev)
        {
            if (e.abs_time - ev->abs_time > width) break;

            if (e.mod != ev->mod)
                coincidences.emplace_back(e,*ev);
        }

        window.erase(ev, window.end());
        e.abs_time += delay;
        window.push_front(e);
    }

    return coincidences;
}

void coincidence_sort_mt (
        std::vector<std::string> fnames,
        std::string output_file
) {
    /*
    std::vector<std::string> fnames = {
        "/opt/acq1/20200311/Derenzo/192.168.1.101.SGL",
        "/opt/acq1/20200311/Derenzo/192.168.1.102.SGL",
        "/opt/acq1/20200311/Derenzo/192.168.2.103.SGL",
        "/opt/acq1/20200311/Derenzo/192.168.2.104.SGL"
    };
    */

    std::ofstream output_file_handle (
            output_file, std::ios::out | std::ios::binary);

    size_t sorter_threads = 8;
    size_t n = fnames.size();

    std::vector<std::mutex> all_lock (n);
    std::vector<std::condition_variable_any> all_cv (n);
    std::vector<std::queue<std::streampos>> all_pos (n);

    std::vector<std::thread> tt_scan;

    // Create time-tag search threads - one per file
    for (size_t i = 0; i < n; i++)
    {
        tt_scan.emplace_back(find_tt_offset,
                             fnames[i],
                             std::ref(all_lock[i]),
                             std::ref(all_cv[i]),
                             std::ref(all_pos[i]));
    }

    std::vector<std::streampos> start_pos(n), end_pos(n);

    // The first value returned is the reset position
    for (size_t i = 0; i < n; i++)
    {
        std::lock_guard<std::mutex> lg(all_lock[i]);

        all_cv[i].wait(
            all_lock[i],
            [&]{ return !all_pos[i].empty(); }
        );

        start_pos[i] = all_pos[i].front();
        all_pos[i].pop();
    }

    // Verify that each thread found a reset
    if (std::any_of(start_pos.begin(), start_pos.end(),
                [](std::streampos i){ return i == -1; }))
    {
        for (auto &th : tt_scan) th.join();
        std::cout << "Failed to find reset" << std::endl;
        return;
    }

    std::vector<CoincidenceData> cd;
    std::deque<std::future<std::vector<CoincidenceData>>> workers;

    bool done = false;
    while (true)
    {
        for (size_t i = 0; i < n; i++)
        {
            std::lock_guard<std::mutex> lg(all_lock[i]);

            all_cv[i].wait(
                all_lock[i],
                [&]{ return !all_pos[i].empty(); }
            );

            end_pos[i] = all_pos[i].front();
            all_pos[i].pop();

            if (end_pos[i] == -1) done = true;
        }

        if (done) break;

        if (workers.size() >= sorter_threads)
        {
            auto new_cd = workers.front().get();
            workers.pop_front();
            CoincidenceData::write(output_file_handle, new_cd);
        }

        workers.push_back(std::async(std::launch::async,
                        &sort_span, fnames, start_pos, end_pos));

        start_pos = end_pos;
    }

    for (auto &scanner : tt_scan)
        scanner.join();

    for (auto &w : workers)
    {
        auto new_cd = w.get();
        CoincidenceData::write(output_file_handle, new_cd);
    }
}
