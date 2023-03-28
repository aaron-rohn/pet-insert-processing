#include "singles.h"

int main()
{
    /*
    uint64_t nev;
    struct SingleData *buf = read_singles(
            "/mnt/acq/20230217/test/192.168.1.101.SGL",
            0, 8ULL*1024*1024*1024, &nev);

    printf("actual events: %llu\n", (unsigned long long)nev);
    free(buf);
    */

    FILE *f = fopen("/mnt/acq/20230217/test/192.168.1.101.SGL", "rb");
    uint64_t tt = 0;

    while (go_to_tt(f, tt))
    {
        printf("%lu\n", tt);
        tt += 100000;
    }
    return 0;
}
