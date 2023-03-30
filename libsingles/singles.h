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
};

typedef struct SinglesReader_ SinglesReader;
typedef struct SingleData_ SingleData;

SinglesReader reader_new(int fd, int is_file);
SinglesReader reader_new_from_file(const char *fname);
off_t reader_pos(SinglesReader *rdr);
int reader_empty(SinglesReader *rdr);

off_t go_to_tt(SinglesReader *rdr, uint64_t value);
SingleData *singles_to_tt(SinglesReader *rdr, uint64_t value, uint64_t *sz);
SingleData *read_singles(const char *fname, off_t start, off_t end, uint64_t *nev);
uint8_t validate(const char *fname);

#ifdef __cplusplus
}

#include <utility>
#include <span>
template <typename T>
class span
{
    public:
    T *data = nullptr;
    const size_t size = 0;
    span(T* data, size_t size): data(data), size(size) {}
    ~span() { if (data) std::free(data); }
    T *begin() const { return data; }
    T *end() const { return data + size; }
};

span<SingleData> span_singles_to_tt(
        SinglesReader *rdr, uint64_t value, uint64_t *sz)
{ 
    SingleData *sgls = singles_to_tt(rdr, value, sz);
    return std::move(span<SingleData>(sgls, *sz));
}

span<SingleData> span_read_singles(
        const char *fname, off_t start, off_t end, uint64_t *nev)
{ 
    SingleData *sgls = read_singles(fname, start, end, nev);
    return std::move(span<SingleData>(sgls, *nev));
}

#endif

#endif
