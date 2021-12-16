#ifndef _SINOGRAM_H
#define _SINOGRAM_H

#include <mutex>
#include <map>
#include <vector>
#include <tuple>
#include <singles.h>
#include <coincidence.h>
#include <Python.h>
#include <json.hpp>
using json = nlohmann::json;

class Geometry
{
    public:
    static const int ncrystals_per_block = 19;

    static const int nblocks_axial = 4;

    // allow for 1 crystal gap between blocks (and at end of scanner)
    static const int nring = nblocks_axial*ncrystals_per_block + nblocks_axial;

    static const int npix  = ncrystals_per_block * Record::nmodules;
    static const int dim_theta = npix;
    static const int dim_r     = npix / 2;

    static const int lut_pix = 512;
    static const int xtal_max = ncrystals_per_block * ncrystals_per_block;

    static constexpr double energy_window = 0.15;
};

class Sinogram: Geometry
{
    std::mutex m;

    public:

    std::vector<int> s;
    Sinogram(): s(std::vector<int> (dim_theta*dim_r, 0)) {};

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

    Michelogram():
            luts(std::vector<std::vector<int>> (Single::nblocks, std::vector<int>(lut_pix*lut_pix, xtal_max))),
            photopeaks(std::vector<std::vector<double>> (Single::nblocks, std::vector<double>(xtal_max, -1))),
            m(std::vector<Sinogram> (nring*nring)) {};

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int h, int v){ return m[v*nring + h]; };

    static std::string find_cfg_file(std::string);
    void load_luts(std::string);
    void load_photopeaks(std::string);

    void add_event(const CoincidenceData&,bool=false);
    void write_to(std::string);
    void read_from(std::string);

    PyObject *to_py_data();
    Michelogram(PyObject*);

    class Iterator
    {
        /*
         * This class is used to iterate over the michelogram segments,
         * starting with segment 0, then +1, -1, +2, -2 ... (nring - 1), -(nring - 1)
         * Within each segment, iteration starts with the plane nearest ring 0
         * and moves towards the maximum positive plane.
         * 'upper' as used here corresponds to negative segments (rd < 0).
         * Horizontal axis is the first ring (ring 0), vert. axis is ring 1
         */

        int r0 = 0, rd = 0;
        bool upper = true;
        Michelogram &m;

        public:

        int h = 0, v = 0;

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

            if (upper)
            {
                h = r0 - rd;
                v = r0;
            }
            else
            {
                h = r0;
                v = r0 - rd;
            }

            return *this;
        }

        Sinogram& operator*() const { return m(h,v); };
        Sinogram* operator->() const { return &m(h,v); };

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
};

std::streampos sort_sinogram_span(std::string,
        std::streampos, std::streampos, int, Michelogram&, std::atomic_bool&);

#endif
