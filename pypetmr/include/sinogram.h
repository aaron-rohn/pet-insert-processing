#ifndef _SINOGRAM_H
#define _SINOGRAM_H

#include <iostream>
#include <mutex>
#include <singles.h>
#include <coincidence.h>
#include <Python.h>
#include <json.hpp>

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

using json = nlohmann::json;
using stype = float;
static const auto npy_type = NPY_FLOAT32;

class Geometry
{
    public:
    static const int ncrystals = 19;
    static const int ncrystals_total = ncrystals * ncrystals;
    static const int nblocks_axial = 4;

    static const int ncrystals_transverse_gap   = 3;
    static const int ncrystals_axial_gap        = 1;

    // allow for 1 crystal gap between blocks (and at end of scanner)
    static const int nring = nblocks_axial*ncrystals + nblocks_axial;

    static const int ncrystals_per_ring = (ncrystals*Record::nmodules) +
        (Record::nmodules*ncrystals_transverse_gap);

    static const int dim_theta = ncrystals_per_ring;
    static const int dim_r     = ncrystals_per_ring / 2;

    static const int lut_dim = 512;
    static constexpr double energy_window = 0.15;
};

class Sinogram: Geometry
{
    std::mutex m;

    public:

    std::vector<stype> s;
    Sinogram(): s(std::vector<stype> (dim_theta*dim_r, 0)) {};

    inline stype& operator() (int theta, int r){ return s[theta*dim_r + r]; };

    static std::tuple<int,int> idx_to_coord(int idx1, int idx2)
    {
        // note that LORs with theta = 0 are tangent to crystal #0
        // theta increases in the same direction as the crystal numbering
        int theta = (idx1 + idx2) % ncrystals_per_ring;
        int r = std::abs(idx1 - idx2);

        if (idx1 + idx2 >= ncrystals_per_ring)
            r = ncrystals_per_ring - r;

        return std::make_tuple(theta, r/2);
    }

    void add_event(int idx1, int idx2)
    {
        auto [theta, r] = idx_to_coord(idx1, idx2);
        std::lock_guard<std::mutex> lck(m);
        (*this)(theta, r)++;
    }

    void write_to(std::ofstream &f)
    { f.write((char*)s.data(), s.size()*sizeof(stype)); };

    void read_from(std::ifstream  &f)
    { f.read((char*)s.data(), s.size()*sizeof(stype)); };
};

class Michelogram: Geometry
{
    const bool flip;
    std::vector<std::vector<int>> luts;
    std::vector<std::vector<double>> photopeaks;
    std::vector<Sinogram> m;

    public:

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int h, int v){ return m[v*nring + h]; };

    static std::string find_cfg_file(std::string);
    void load_luts(std::string);
    void load_photopeaks(std::string);

    void add_event(const CoincidenceData&);
    void write_to(std::string);
    void read_from(std::string);

    PyObject *to_py_data();

    Michelogram(PyObject*);

    Michelogram(bool flip = false):
        flip(flip),
        luts(std::vector<std::vector<int>> (Single::nblocks, std::vector<int>(lut_dim*lut_dim, ncrystals_total))),
        photopeaks(std::vector<std::vector<double>> (Single::nblocks, std::vector<double>(ncrystals_total, -1))),
        m(std::vector<Sinogram> (nring*nring)) {};

    std::streampos sort_span(std::string, std::streampos, std::streampos);

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

            h = r0;
            v = r0 - rd;
            if (upper) std::swap(h,v);
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

#endif
