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

Single::Single(uint8_t data[])
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
    //abs_time = time;
    abs_time = 0;
}

void Single::set_abs_time(TimeTag &last_tt)
{
    abs_time = last_tt.value * CLK_PER_TT + time;
}

uint64_t SinglesReader::length()
{
    uint64_t file_length, file_elems;
    fseek(f, 0L, SEEK_END);
    file_length = ftell(f);
    rewind(f);
    nsingles = 0;
    ntimetag = 0;

    file_elems = file_length / EV_SIZE;
    return file_elems;
}

void SinglesReader::set_entry()
{
    entry_is_single = DATA_FLAG(data);
    mod = DATA_MOD(data);
    if (entry_is_single) {
        entry.single = Single(data);
        nsingles++;
    } else {
        entry.timetag = TimeTag(data);
        ntimetag++;
    }
}

uint64_t SinglesReader::find_rst()
{
    uint64_t offset;
    while ((offset = read()))
    {
        if (!entry_is_single && entry.timetag.value == 0) break;
    }
    return offset;
}

uint64_t SinglesReader::read()
{
    if (fread(data, 1, EV_SIZE, f) != EV_SIZE) return 0;

    // ensure alignment of data stream
    while (!BYTE_IS_HEADER(data[0]))
    {
        size_t n = EV_SIZE;
        for (size_t i = 1; i < EV_SIZE; i++)
        {
            if (BYTE_IS_HEADER(data[i]))
            {
                std::memmove(data, data + i, EV_SIZE - i);
                n = i;
                break;
            }
        }

        if (fread(data + (EV_SIZE - n), 1, n, f) != n) return 0;
    }

    set_entry();
    return ftello(f);
}

