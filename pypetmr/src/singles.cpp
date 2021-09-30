#include "singles.h"

TimeTag::TimeTag(uint8_t data[])
{
    // Time tag
    // CRC | f |    b   |                     0's                    |             TT
    // { 5 , 1 , 2 }{ 4 , 4 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }{ 8 }
    //       0          1      2    3    4    5    6    7    8    9   10   11   12   13   14   15

    mod = DATA_MOD(data);
    uint64_t upper = (data[10] << 16) | (data[11] << 8) | data[12];
    uint64_t lower = (data[13] << 16) | (data[14] << 8) | data[15];
    value = (upper << 24) | lower;
}

Single::Single(uint8_t data[], TimeTag &tt)
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

    block = DATA_BLK(data);
    time = ((data[13] << 16) | (data[14] << 8) | data[15]) & 0xFFFFF;
    abs_time = tt.value * CLK_PER_TT + time;
}

SinglesReader::SinglesReader (const char *fname):
    tt(nblocks)
{
    f = fopen(fname, "rb");
    fseek(f, 0L, SEEK_END);
    file_size = ftell(f);
    file_elems = file_size / event_size;
    rewind(f);
}

uint64_t SinglesReader::find_rst()
{
    while (read())
    {
        if (!is_single && timetag.value == 0) break;
    }
    return offset;
}

uint64_t SinglesReader::read()
{
    if (fread(data, 1, event_size, f) != event_size)
    {
        offset = 0;
        return offset;
    }

    // ensure alignment of data stream
    while (!BYTE_IS_HEADER(data[0]))
    {
        size_t n = event_size;
        for (size_t i = 1; i < event_size; i++)
        {
            if (BYTE_IS_HEADER(data[i]))
            {
                std::memmove(data, data + i, event_size - i);
                n = i;
                break;
            }
        }

        if (fread(data + (event_size - n), 1, n, f) != n)
        {
            offset = 0;
            return offset;
        }
    }

    is_single = DATA_FLAG(data);
    mod = DATA_MOD(data);

    if (is_single)
    {
        single = Single(data, tt[mod]);
        nsingles++;
    }
    else
    {
        TimeTag new_tt = TimeTag(data);
        tt_aligned = (tt[mod].value + 1 == new_tt.value);
        timetag = new_tt;
        tt[mod] = new_tt;
        ntimetag++;
    }

    offset = ftello(f);
    return offset;
}

std::vector<std::deque<Single>> 
SinglesReader::go_to_tt(uint64_t target)
{
    std::vector<std::deque<Single>> events (SinglesReader::nblocks);

    while (read() && (!tt_aligned || timetag.value < target))
    {
        if (is_single)
        {
            events[single.block].push_back(single);
        }
    }

    return events;
}
