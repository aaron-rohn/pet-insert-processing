#ifndef _SINGLES_PACK_H
#define _SINGLES_PACK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NMODULES 16
#define CLK_PER_TT 800000UL
#define MODULE(blk) (blk >> 2)
#define MODULE_ABOVE(mod) ((mod + 1) % 16)
#define MODULE_BELOW(mod) ((mod + 15) % 16)
#define VALID_MODULE(a,b) ((a != b) && (a != MODULE_ABOVE(b)) && (a != MODULE_BELOW(b)))

struct SingleData
{
    uint8_t block;
    uint16_t eR, eF;
    uint16_t x, y;
    uint64_t abstime;
};

#ifdef __cplusplus
extern "C" {
#endif

off_t go_to_tt(FILE *f, uint64_t value);
struct SingleData *read_singles(const char *fname, off_t start, off_t end, uint64_t *nev);
uint8_t validate(const char *fname);

#ifdef __cplusplus
}
#endif

#endif
