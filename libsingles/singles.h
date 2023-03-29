#ifndef _SINGLES_PACK_H
#define _SINGLES_PACK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NMODULES 16
#define CLK_PER_TT 800000UL
#define MODULE(blk) ((blk >> 2) & 0xF)
#define MODULE_ABOVE(mod) ((mod + 1) % NMODULES)
#define MODULE_BELOW(mod) ((mod + NMODULES - 1) % NMODULES)
#define VALID_MODULE(a,b) ((a != b) && (a != MODULE_ABOVE(b)) && (a != MODULE_BELOW(b)))

struct __attribute__((packed)) SingleData
{
    // pack to 16 bytes
    uint32_t block  : 8;
    uint32_t x      : 12;
    uint32_t y      : 12;
    uint16_t eR, eF;
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
