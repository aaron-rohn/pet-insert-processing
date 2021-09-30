#include "coincidence.h"

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

    total_counts += new_coins.size();
    window.erase(ev, window.end());
    new_ev.abs_time += delay;
    window.push_front(new_ev);
    return new_coins;
}
