#include "coincidence.h"
#include <cmath>

Coincidence::Coincidence(Single ev1, Single ev2) {
    if (ev1.block < ev2.block) {
        a = ev1;
        b = ev2;
    } else {
        a = ev2;
        b = ev1;
    }
}

std::deque<Coincidence> CoincidenceSorter::add_event(Single new_ev)
{
    std::deque<Coincidence> new_coins;
    auto ev = window.begin();
    for (; ev != window.end(); ++ev)
    {
        if (new_ev.abs_time - ev->abs_time > width) break;

        if (new_ev.mod != ev->mod)
            new_coins.emplace_back(new_ev,*ev);
    }

    counts += new_coins.size();
    window.erase(ev, window.end());
    new_ev.abs_time += delay;
    window.push_front(new_ev);
    return new_coins;
}

void CoincidenceSorter::write_events(const std::deque<Coincidence> &coincidences)
{
    if (output_file)
    {
        uint16_t data[vals_per_ev];
        for (const auto &c : coincidences)
        {
            auto sda = SingleData(c.a), sdb = SingleData(c.b);

            int16_t tdiff = (int64_t)c.a.abs_time - c.b.abs_time;
            uint16_t xa = std::round(sda.x * scale);
            uint16_t ya = std::round(sda.y * scale);
            uint16_t xb = std::round(sdb.x * scale);
            uint16_t yb = std::round(sdb.y * scale);

            data[0] = (c.a.block << 8) | c.b.block;
            data[1] = *((uint16_t*)&tdiff);

            data[2] = sda.e1;
            data[3] = sda.e2;
            data[4] = sdb.e1;
            data[5] = sdb.e2;

            data[6] = xa;
            data[7] = ya;
            data[8] = xb;
            data[9] = yb;

            output_file.write((char*)data, sizeof(data));
        }
    }
}
