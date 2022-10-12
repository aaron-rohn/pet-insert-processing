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

template<typename T> using vec = std::vector<T>;

class Geometry
{
    public:
    static const int ncrystals = 19;
    static const int nblocks_axial = 4;
    static const int ncrystals_total = ncrystals * ncrystals;

    static const int ncrystals_axial_gap = 1;
    static const int nring = nblocks_axial *
        (ncrystals + ncrystals_axial_gap);

    static const int ncrystals_transverse_gap = 0;
    static const int ncrystals_per_ring = Record::nmodules *
        (ncrystals + ncrystals_transverse_gap);

    static const int dim_theta_full = ncrystals_per_ring;
    static const int dim_theta_half = ncrystals_per_ring/2;
    static const int dim_r          = ncrystals_per_ring;

    static constexpr double energy_window = 0.2;

    static inline int ring(int blk, int xtal)
    {
        int row = ncrystals - (xtal / ncrystals) - 1;
        int blk_ax = blk % nblocks_axial;
        return row + (ncrystals + ncrystals_axial_gap)*blk_ax;
    }

    static inline int idx(int blk, int xtal)
    {
        // crystals are indexed in the opposite direction from increasing module number
        int col = xtal % ncrystals;
        int mod = blk >> 2;
        return col + (ncrystals + ncrystals_transverse_gap)*mod;
    }
};

class CrystalLookupTable
{
    std::vector<std::vector<int>> luts;

    public:
    static const int lut_dim = 512;

    CrystalLookupTable():
        luts(std::vector<std::vector<int>> (
                    Single::nblocks,
                    std::vector<int>(
                        lut_dim*lut_dim,
                        Geometry::ncrystals_total))) {};

    inline int& operator() (size_t blk, size_t pos_y, size_t pos_x)
    { return luts[blk][pos_y*lut_dim + pos_x]; }

    void load(std::string);
};

class PhotopeakLookupTable
{
    vec<vec<double>> photopeaks;
    vec<vec<vec<double>>> doi;

    public:
    const double energy_window = Geometry::energy_window;

    PhotopeakLookupTable():
        photopeaks(vec<vec<double>> (
                    Single::nblocks,
                    vec<double>(Geometry::ncrystals_total, -1))),
        doi(vec<vec<vec<double>>> (
                    Single::nblocks,
                    vec<vec<double>>(
                        Geometry::ncrystals_total,
                        vec<double>()))) {};

    void load(std::string);
    static std::string find_cfg_file(std::string);

    inline bool in_window(size_t blk, size_t xtal, double e)
    {
        double th = photopeaks[blk][xtal];
        return (th < 0) || ((e > (1.0-energy_window)*th) &&
                            (e < (1.0+energy_window)*th));
    }

    inline int doi_window(size_t blk, size_t xtal, double val)
    {
        const auto &doi_vals = doi[blk][xtal];
        for (size_t i = 0; i < doi_vals.size(); i++)
            if (val > doi_vals[i]) return i;
        return doi_vals.size();
    }
};

class Sinogram: Geometry
{
    std::mutex m;

    public:

    std::vector<stype> s;
    Sinogram(int dt):
        s(std::vector<stype> (dt*dim_r, 0)) {};
    Sinogram(const Sinogram &other): m(), s(other.s) {};

    inline stype& operator() (int theta, int r){ return s[theta*dim_r + r]; };

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
    std::vector<Sinogram> m;

    public:

    CrystalLookupTable lut;
    PhotopeakLookupTable ppeak;

    void cfg_load(std::string cfg_dir)
    {
        lut.load(cfg_dir);
        ppeak.load(cfg_dir);
    }

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int h, int v){ return m[v*nring + h]; };

    std::tuple<bool, int, int, int, int, int, int> event_to_coords(
            const CoincidenceData&, double);

    void add_event(const CoincidenceData&, double=1.0);
    void write_event(std::ofstream&, const CoincidenceData&, double=1.0);
    void write_to(std::string);
    void read_from(std::string);

    PyObject *to_py_data();
    Michelogram(PyObject*);
    Michelogram(int dt):
        m(std::vector<Sinogram> (nring*nring, Sinogram(dt))) {};

    std::streampos sort_span(
            std::string, uint64_t, uint64_t,
            double const*, uint64_t const*, size_t);

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
