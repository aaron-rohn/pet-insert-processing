#ifndef _SINOGRAM_H
#define _SINOGRAM_H

#include <iostream>
#include <mutex>
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include "constants.h"
#include "singles.h"
#include "coincidence.h"

using stype = float;
static const auto npy_type = NPY_FLOAT32;

class Sinogram
{
    std::mutex m;

    public:

    std::vector<stype> s;
    Sinogram(size_t dt): s(std::vector<stype> (dt*Geometry::dim_r, 0)) {};
    Sinogram(const Sinogram &other): m(), s(other.s) {};

    inline stype& operator() (int theta, int r){ return s[theta*Geometry::dim_r + r]; };

    static std::tuple<int,int> idx_to_coord(int idx1, int idx2)
    {
        // note that LORs with theta = 0 are tangent to crystal #0
        // theta increases in the same direction as the crystal numbering
        int theta = idx1 + idx2;
        int r = idx2 - idx1;

        // Reduce the number of proj. angles by 1/2 and
        // allow 2x sampling of FOV
        return std::make_tuple(theta/2, r);
    }

    void add_event(int idx1, int idx2)
    {
        auto [theta, r] = idx_to_coord(idx1, idx2);

        // trim outermost blocks from transverse dimension of sinogram
        r -= Geometry::ncrystals;
        if (r < 0 || r > Geometry::dim_r) return;

        std::lock_guard<std::mutex> lck(m);
        (*this)(theta, r)++;
    }

    void write_to(std::ofstream &f)
    { f.write((char*)s.data(), s.size()*sizeof(stype)); };

    void read_from(std::ifstream  &f)
    { f.read((char*)s.data(), s.size()*sizeof(stype)); };

    static inline int ring(int blk, int xtal)
    {
        int row = Geometry::ncrystals - (xtal / Geometry::ncrystals) - 1;
        int blk_ax = blk % Geometry::nblocks_axial;
        return row + (Geometry::ncrystals + Geometry::ncrystals_axial_gap)*blk_ax;
    }

    static inline int idx(int blk, int xtal)
    {
        int col = xtal % Geometry::ncrystals;
        int mod = blk >> 2;
        return col + (Geometry::ncrystals + Geometry::ncrystals_transverse_gap)*mod;
    }
};

class Michelogram
{
    std::vector<Sinogram> m;
    PyArrayObject *photopeaks, *doi, *lut;

    public:

    const size_t max_doi = Geometry::ndoi;
    const double energy_scale = 63.0;

    bool loaded() { return photopeaks && doi && lut; }

    inline int lut_lookup(int blk, int scale_idx, int y, int x) const
    { return *(int*)PyArray_GETPTR4(lut, blk, scale_idx, y, x); }

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int h, int v){ return m[v*Geometry::nring + h]; };

    int energy_window(size_t, size_t, size_t, double) const;
    int doi_window(size_t, size_t, size_t, double) const;
    void write_to(std::string);
    void read_from(std::string);
    ListmodeData event_to_coords(const CoincidenceData&, size_t) const;
    std::streampos sort_span(std::string, std::streampos, std::streampos, bool, bool);
    FILE *encode_span (std::string, std::streampos, std::streampos, int) const;

    PyObject *to_py_data();
    Michelogram(PyObject*);

    Michelogram(size_t dt, size_t max_doi = Geometry::ndoi,
            PyArrayObject *lut = NULL,
            PyArrayObject *photopeaks = NULL,
            PyArrayObject *doi = NULL):
        m(std::vector<Sinogram> (Geometry::nring*Geometry::nring, Sinogram(dt))),
        photopeaks(photopeaks), doi(doi), lut(lut),
        max_doi(max_doi) {};

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
            if (++r0 == Geometry::nring)
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
    Iterator end() { return Iterator(Geometry::nring, *this); };
};

#endif
