#include "singles.h"

int main()
{
    uint64_t nev = 0;
    SingleData *buf = read_singles(
            "/mnt/acq/20230217/test/192.168.1.101.SGL",
            0, -1, &nev);

    printf("actual events: %llu\n", (unsigned long long)nev);
    free(buf);

    SinglesReader rdr = reader_new_from_file("/mnt/acq/20230217/test/192.168.1.101.SGL");
    uint64_t tt = 0;

    while (go_to_tt(&rdr, tt))
    {
        printf("%lu\n", tt);
        tt += 100000;
    }
    return 0;
}
