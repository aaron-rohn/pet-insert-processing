#ifndef _SINOGRAM_H
#define _SINOGRAM_H

#include <map>
#include <vector>
#include <tuple>
#include <coincidence.h>

#include <json.hpp>
using json = nlohmann::json;

class Geometry
{
    public:
    static const int ncrystals_per_block = 19;

    static const int nblocks_per_ring = 16;
    static const int nblocks_axial = 4;
    static const int nring = nblocks_axial * ncrystals_per_block;

    static const int npix  = ncrystals_per_block * nblocks_per_ring;
    static const int dim_theta = npix;
    static const int dim_r     = npix / 2;

    static const int lut_pix = 512;
    static const int xtal_max = ncrystals_per_block * ncrystals_per_block;

    static constexpr double energy_window = 0.15;
};

class Sinogram: Geometry
{
    std::vector<int> s;

    public:

    Sinogram(): s(std::vector<int>(dim_theta*dim_r, 0)) {};
    inline int& operator() (int theta, int r){ return s[theta*dim_r + r]; };
    void add_event(int,int);

    void write_to(std::ofstream &f)
    { f.write((char*)s.data(), s.size()*sizeof(int)); };

    void read_from(std::ifstream  &f)
    { f.read((char*)s.data(), s.size()*sizeof(int)); };
};

class Michelogram: Geometry
{
    std::vector<std::vector<int>> luts;
    std::vector<std::vector<double>> photopeaks;
    std::vector<Sinogram> m;

    public:

    class Iterator
    {
        /*
         * This class is used to iterate over the michelogram segments,
         * starting with segment 0, then +1, -1, +2, -2 ... (nring - 1), -(nring - 1)
         * Within each segment, iteration starts with the plane nearest ring 0
         * and moves towards the maximum positive plane.
         * 'upper' as used here corresponds to negative segments (rd < 0).
         */

        int r0 = 0, rd = 0;
        bool upper = true;
        Michelogram &m;

        public:

        Iterator(int rd, Michelogram &m):
            rd(rd), m(m) {};

        Iterator& operator++()
        {
            // Increment horiz. coord and check if segment finished
            if (++r0 == nring)
            {
                // If 'upper' segment is finished, increment ring diff.
                if (upper) ++rd;

                // Either move to the upper segment, or next lower segment
                // horiz. coordinate starts at the value of the ring diff.
                upper = !upper;
                r0 = rd;
            }
            return *this;
        }

        // r0 is the horiz. coord in the Michelogram, r1 = r0 - rd is the vert. coord
        Sinogram& operator*() const
        { return upper ? m(r0 - rd, r0) : m(r0, r0 - rd); };

        // Compare ring difference - only valid to determine the end iterator
        friend bool operator!=(Iterator &a, Iterator &b)
        { return a.rd != b.rd; };

        using iterator_category = std::forward_iterator_tag;
        using difference_type   = int;
        using value_type        = Sinogram;
        using pointer           = Sinogram*;
        using reference         = Sinogram&;
    };

    // start iteration with segment 0
    Iterator begin() { return Iterator(0, *this); };

    // maximum valid segment is nring-1
    Iterator end() { return Iterator(nring, *this); };

    Michelogram(std::string base_dir):
        luts(load_luts(base_dir)),
        photopeaks(load_photopeaks(base_dir)),
        m(std::vector<Sinogram>(nring*nring)) {}

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int r0, int r1){ return m[r1*nring + r0]; };

    static std::vector<std::vector<double>> load_photopeaks(std::string);
    static std::vector<std::vector<int>> load_luts(std::string);

    void add_event(const CoincidenceData&);
    void write_to(std::string);
    void read_from(std::string);
};

#endif
