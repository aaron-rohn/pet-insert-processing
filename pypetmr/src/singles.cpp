#include "singles.h"

TimeTag::TimeTag(uint8_t data[])
{
    // Time tag
    // CRC | f |    b   |                     0's                    |             TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }
    //       0          1      2    3    4    5    6    7    8    9   10   11   12   13   14   15

    mod = Single::get_module(data);
    uint64_t upper = (data[10] << 16) | (data[11] << 8) | data[12];
    uint64_t lower = (data[13] << 16) | (data[14] << 8) | data[15];
    value = (upper << 24) | lower;
}

Single::Single(uint8_t data[], const TimeTag &tt)
{
    // Single event
    // CRC | f |    b   |   E1   |    E2   |   E3    |   E4   |   E5    |   E6   |   E7    |   E8   |       TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }{ 4 , 4 }{ 8 }{ 8 }
    //       0          1      2    3      4      5    6      7      8    9     10     11   12     13     14   15

    // Front energies
    energies[0] = ((data[1] << 8) | data[2]) & 0xFFF;   // A
    energies[1] = (data[3] << 4) | (data[4] >> 4);      // B
    energies[2] = ((data[4] << 8) | data[5]) & 0xFFF;   // C
    energies[3] = (data[6] << 4) | (data[7] >> 4);      // D

    // Rear energies
    energies[4] = ((data[7] << 8) | data[8]) & 0xFFF;   // E
    energies[5] = (data[9] << 4) | (data[10] >> 4);     // F
    energies[6] = ((data[10] << 8) | data[11]) & 0xFFF; // G
    energies[7] = (data[12] << 4) | (data[13] >> 4);    // H

    block = get_block(data);
    mod = get_module(data);
    time = ((data[13] << 16) | (data[14] << 8) | data[15]) & 0xFFFFF;
    abs_time = tt.value * TimeTag::clks_per_tt + time;
}

bool Single::is_header(uint8_t byte) {
    return (byte >> 3) == 0x1F;
}

bool Single::is_single(uint8_t data[]) {
    return data[0] & 0x4;
}

uint8_t Single::get_block(uint8_t data[]) {
    return ((data[0] << 4) | (data[1] >> 4)) & 0x3F;
}

uint8_t Single::get_module(uint8_t data[]) {
    return ((data[0] << 2) | (data[1] >> 6)) & 0xF;
}

SinglesReader::SinglesReader (std::string fname):
    fname(fname),
    f(fname, std::ios::binary),
    tt(nmodules)
{
    if (!f.good()) return;

    f.seekg(0, std::ios::end);
    file_size = f.tellg();
    file_elems = file_size / event_size;
    f.seekg(0, std::ios::beg);

    std::cout << fname << ": " <<
        file_elems << " entries" << std::endl;
}

void SinglesReader::align()
{
    while (f.good() && !Single::is_header(data[0]))
    {
        size_t n = event_size;
        for (size_t i = 1; i < event_size; i++)
        {
            if (Single::is_header(data[i]))
            {
                std::cout << fname << ": Realign data stream (0x" <<
                    std::hex << f.tellg() << std::dec << ")\n";

                std::memmove(data, data + i, event_size - i);
                n = i;
                break;
            }
        }
        f.read((char*)(data + (event_size - n)), n);
    }
}

bool SinglesReader::read()
{
    f.read((char*)data, event_size);
    align();

    is_single = Single::is_single(data);
    mod = Single::get_module(data);

    if (is_single)
    {
        single = Single(data, tt[mod]);
        nsingles++;
    }
    else
    {
        TimeTag new_tt (data);
        tt_aligned = (tt[mod].value + 1 == new_tt.value);
        tt[mod] = new_tt;
        ntimetag++;
    }

    return f.good();
}

bool SinglesReader::find_rst()
{
    while (read())
    {
        if (!is_single && tt[mod].value == 0) break;
    }

    std::cout << fname << ": Found reset at offset 0x" <<
        std::hex << f.tellg() << std::dec << std::endl;

    return f.good();
}

std::vector<std::deque<Single>> 
SinglesReader::go_to_tt(uint64_t target)
{
    std::vector<std::deque<Single>> events (nmodules);

    while (read())
    {
        if (is_single)
        {
            events[single.mod].push_back(single);
        }
        else
        {
            if (tt_aligned && tt[mod].value >= target) break;
        }
    }

    return events;
}
