#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "singles.h"

#define VALID_HEADER 0x1F
#define FLOOD_COORD_SCALE 511
#define MAX_BUF_SIZE (16ULL*1024*1024*1024)

struct __attribute__((packed, scalar_storage_order("big-endian"))) Single
{
    uint8_t header      : 5;
    uint8_t flag        : 1;
    uint8_t block       : 6;
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
    uint8_t header  : 5;
    uint8_t flag    : 1;
    uint8_t block   : 6;
    uint8_t         : 4;
    uint64_t        : 64;
    uint64_t coarse_time : 48;
};

union Record
{
    struct Single sgl;
    struct TimeTag tt;
};

union Record *read_record(char **bstart, char *bend)
{
    static const size_t incr = sizeof(union Record);
    union Record *d = (union Record*)(*bstart);
    while (d->sgl.header != VALID_HEADER && *bstart < bend)
    {
        (*bstart)++;
        d = (union Record*)(*bstart);
    }

    if (*bstart + incr > bend)
        return NULL;

    (*bstart) += incr;
    return d;
}

off_t go_to_tt(FILE *f, uint64_t value)
{
    union Record *d;
    uint64_t last_tt_value = 0;
    int synced = 0;

    char buf[1024*8];
    char *bstart = buf, *bend = buf;

    while (1)
    {
        size_t occ = bend - bstart;
        if (occ > 0) memmove(buf, bstart, occ);

        bstart = buf;
        bend = buf + occ;
        bend += fread(bend, 1, sizeof(buf) - occ, f);

        if (bstart == bend) break;

        while ((d = read_record(&bstart, bend)) != NULL)
        {
            if (!d->sgl.flag)
            {
                synced = d->tt.coarse_time == (last_tt_value + 1);
                last_tt_value = d->tt.coarse_time;

                if ((value == 0 && d->tt.coarse_time == 0) ||
                        (synced && value > 0 && d->tt.coarse_time >= value))
                    return ftello(f) - (bend - bstart);
            }
        }
    }

    return 0;
}

void to_single_data(struct Single *d, struct TimeTag *tt, struct SingleData *sd)
{
    sd->block = d->block;
    sd->eR = d->a_rear + d->b_rear + d->c_rear + d->d_rear;
    sd->eF = d->a_front + d->b_front + d->c_front + d->d_front;

    double xF = ((double)(d->a_front + d->b_front)) / sd->eF;
    double yF = ((double)(d->a_front + d->d_front)) / sd->eF;
    double xR = ((double)(d->a_rear + d->b_rear)) / sd->eR;
    double yR = ((double)(d->a_rear + d->d_rear)) / sd->eR;

    sd->x = round(xR * FLOOD_COORD_SCALE);
    sd->y = round((yF + yR) / 2.0 * FLOOD_COORD_SCALE);
    sd->abstime = tt->coarse_time * CLK_PER_TT + d->fine_time;
}

struct SingleData *read_singles(const char *fname, off_t start, off_t end, uint64_t *nev)
{
    FILE *f = fopen(fname, "rb");

    if (start < 0) start = 0;

    if (end <= start)
    {
        fseeko(f, 0, SEEK_END);
        end = ftello(f);
    }

    off_t span = end - start;
    if (span > MAX_BUF_SIZE)
    {
        fprintf(stderr, "Truncate singles buffer length to 16GiB\n");
        span = MAX_BUF_SIZE;
    }

    fseeko(f, start, SEEK_SET);
    char *data = malloc(span);
    span = fread(data, 1, span, f);
    fclose(f);

    struct TimeTag tt[NMODULES];
    uint64_t nev_approx = span / sizeof(union Record);
    struct SingleData* buf = calloc(nev_approx, sizeof(struct SingleData));
    union Record *d;
    *nev = 0;

    char *bstart = data, *bend = data + span;
    while ((d = read_record(&bstart, bend)) != NULL)
    {
        if (d->sgl.flag)
        {
            to_single_data(&d->sgl, &tt[MODULE(d->sgl.block)], &buf[*nev]);
            (*nev)++;
        }
        else
        {
            tt[MODULE(d->tt.block)] = d->tt;
        }
    }

    free(data);
    return buf;
}

uint8_t validate(const char *fname)
{
    FILE *f = fopen(fname, "rb");
    union Record *d;

    char buf[1024*8];
    char *bstart = buf, *bend = buf;

    uint8_t flags = 0, flags_valid = 0x1F;
    const int nmod = 4;

    while (1)
    {
        size_t occ = bend - bstart;
        if (occ > 0) memmove(buf, bstart, occ);

        bstart = buf;
        bend = buf + occ;
        bend += fread(bend, 1, sizeof(buf) - occ, f);

        if (bstart == bend) break;

        while ((d = read_record(&bstart, bend)) != NULL)
        {
            if (!d->sgl.flag)
            {
                if (d->tt.coarse_time == 0) flags |= (1 << 4);
                flags |= (1 << (MODULE(d->tt.block) % nmod));
                if (flags == flags_valid) return flags;
            }
        }
    }

    return flags;
}
