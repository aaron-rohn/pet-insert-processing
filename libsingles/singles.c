#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "singles.h"

#define VALID_HEADER 0x1F
#define FLOOD_COORD_SCALE 511

struct __attribute__((packed, scalar_storage_order("big-endian"))) Single
{
    uint16_t header     : 5;
    uint16_t flag       : 1;
    uint16_t block      : 6;
    uint16_t d_rear     : 12;
    uint16_t c_rear     : 12;
    uint16_t b_rear     : 12;
    uint16_t a_rear     : 12;
    uint16_t d_front    : 12;
    uint16_t c_front    : 12;
    uint16_t b_front    : 12;
    uint16_t a_front    : 12;
    uint32_t fine_time  : 20;
};

struct __attribute__((packed, scalar_storage_order("big-endian"))) TimeTag
{
    uint16_t header : 5;
    uint16_t flag   : 1;
    uint16_t block  : 6;
    uint16_t        : 4;
    uint64_t        : 64;
    uint64_t coarse_time : 48;
};

union Record_
{
    struct Single sgl;
    struct TimeTag tt;
};

struct Reader
{
    FILE *f;
    char buf[1024*1024];
    char *start, *end;
    off_t pos;
};

typedef union Record_ Record;

void to_single_data(struct Single *d, struct TimeTag *tt, struct SingleData *sd)
{
    sd->block = d->block;
    sd->abstime = tt->coarse_time * CLK_PER_TT + d->fine_time;

    sd->eR = d->a_rear + d->b_rear + d->c_rear + d->d_rear;
    sd->eF = d->a_front + d->b_front + d->c_front + d->d_front;

    double xF = ((double)(d->a_front + d->b_front)) / sd->eF;
    double yF = ((double)(d->a_front + d->d_front)) / sd->eF;
    double xR = ((double)(d->a_rear + d->b_rear)) / sd->eR;
    double yR = ((double)(d->a_rear + d->d_rear)) / sd->eR;

    sd->x = round(xR * FLOOD_COORD_SCALE);
    sd->y = round((yF + yR) / 2.0 * FLOOD_COORD_SCALE);
}

Record *read_record(char **start, char *end)
{
    // rec and start both point to the cursor
    Record **rec = (Record**)start;

    // ensure byte-alignment of the single event
    while (*start < end && (*rec)->sgl.header != VALID_HEADER)
        (*start)++; // advance cursor by 1 byte until reaching header

    // ensure that the buffer has enough data
    // return the event under the cursor, then advance the cursor by 1 event
    return (*start + sizeof(Record) > end) ? NULL : (*rec)++;
}

struct Reader new_reader(FILE *f)
{
    struct Reader rdr = {.f = f};
    rdr.start = rdr.end = rdr.buf;
    rdr.pos = ftello(rdr.f);
    return rdr;
}

Record *reader_read(struct Reader *rdr)
{
    do
    {
        Record *d = read_record(&(rdr->start), rdr->end);
        if (d) return d;

        // insufficient data left in buffer

        // move any 'tail' to the start of the buffer
        size_t occ = rdr->end - rdr->start;
        if (occ > 0) memmove(rdr->buf, rdr->start, occ);

        // re-fill the buffer
        rdr->start = rdr->buf;
        rdr->end   = rdr->buf + occ;
        rdr->end   += fread(rdr->end, 1, sizeof(rdr->buf) - occ, rdr->f);
        rdr->pos   = ftello(rdr->f);
    }
    while (rdr->start != rdr->end); // end when no data left in file

    return NULL;
}

off_t reader_pos(struct Reader *rdr)
{
    return rdr->pos - (rdr->end - rdr->start);
}

off_t go_to_tt(FILE *f, uint64_t value)
{
    Record *d;
    uint64_t last_tt_value = 0;
    int synced = 0;

    struct Reader rdr = new_reader(f);
    while ((d = reader_read(&rdr)))
    {
        if (d->sgl.flag) continue;

        synced = d->tt.coarse_time == (last_tt_value + 1);
        last_tt_value = d->tt.coarse_time;

        if ((value == 0 && d->tt.coarse_time == 0) ||
                (synced && value > 0 && d->tt.coarse_time >= value))
            return reader_pos(&rdr);
    }

    return 0;
}

struct SingleData *read_singles(
        const char *fname,
        off_t start, off_t end, uint64_t *nev
) {
    FILE *f = fopen(fname, "rb");

    if (start < 0) start = 0;

    if (end <= start)
    {
        fseeko(f, 0, SEEK_END);
        end = ftello(f);
    }

    fseeko(f, start, SEEK_SET);

    // Get the lesser of the specified max events or the file size
    // Note that file size includes timetags, so it's only an approximate
    // estimate of the number of singles
    uint64_t approx = (end - start) / sizeof(Record);
    uint64_t len = *nev ?: approx; // input value of nev is the max events
    len = len < approx ? len : approx;

    struct SingleData *buf = calloc(len, sizeof(struct SingleData));

    *nev = 0;
    Record *d;
    struct TimeTag tt[NMODULES];

    struct Reader rdr = new_reader(f);
    while ((d = reader_read(&rdr)) && (*nev < len) && (reader_pos(&rdr) < end))
    {
        if (d->sgl.flag)
        {
            to_single_data(&d->sgl, &tt[MODULE(d->sgl.block)], &buf[(*nev)++]);
        }
        else
        {
            tt[MODULE(d->tt.block)] = d->tt;
        }
    }

    fclose(f);
    return buf;
}

uint8_t validate(const char *fname)
{
    Record *d;

    const int nmod = 4;
    const uint8_t all_valid = 0x1F;
    uint8_t flags = 0;

    struct Reader rdr = {.f = fopen(fname, "rb")};
    rdr.start = rdr.end = rdr.buf;

    while ((d = reader_read(&rdr)) && (flags != all_valid))
    {
        if (d->sgl.flag) continue;

        if (d->tt.coarse_time == 0)
            flags |= (1 << 4); // set reset bit

        // set per-module 0-3 bit
        flags |= (1 << (MODULE(d->tt.block) % nmod));
    }

    fclose(rdr.f);
    return flags;
}
