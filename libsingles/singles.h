#ifndef _SINGLES_PACK_H
#define _SINGLES_PACK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#define NMODULES 16
#define CLK_PER_TT 800000UL
#define MODULE(blk) ((blk >> 2) & 0xF)
#define MODULE_ABOVE(mod) ((mod + 1) % NMODULES)
#define MODULE_BELOW(mod) ((mod + NMODULES - 1) % NMODULES)
#define VALID_MODULE(a,b) ((a != b) && (a != MODULE_ABOVE(b)) && (a != MODULE_BELOW(b)))

#ifdef __cplusplus
extern "C" {
#endif

struct __attribute__((packed)) SingleData_
{
    // pack to 16 bytes
    uint32_t block  : 8;
    uint32_t x      : 12;
    uint32_t y      : 12;
    uint16_t eR, eF;
    uint64_t abstime;
};

struct SinglesReader_
{
    int fd, is_file, finished;
    char buf[1024*1024];
    char *start, *end;
    off_t fpos;
    uint64_t tt[NMODULES];
};

typedef struct SinglesReader_ SinglesReader;
typedef struct SingleData_ SingleData;

// Functions for working with singles reader objects
SinglesReader reader_new(int fd, int is_file);
SinglesReader reader_new_from_file(const char *fname);
off_t reader_pos(SinglesReader *rdr);
int reader_empty(SinglesReader *rdr);

// Advance a reader object to a specified timetag value
off_t go_to_tt(SinglesReader *rdr, uint64_t value);

// Advance a reader object to a specified timetag value,
// and return all single events encountered. Allow for a hint
// at the expected number of singles, for pre-allocation
SingleData *singles_to_tt(SinglesReader *rdr, uint64_t value, uint64_t *sz);

// Read singles from offset *start* to *end*. Note that parameter end
// is an in/out-parameter, to allow seeking back to a location that
// has proper byte-header alignment.
SingleData *read_singles(const char *fname, off_t start, off_t *end, uint64_t *nev);

// Validate that a singles file contains a reset and data from four modules
uint8_t validate(const char *fname);

#ifdef __cplusplus
}
#endif

#endif
