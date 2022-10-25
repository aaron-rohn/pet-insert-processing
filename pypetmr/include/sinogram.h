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

#include <opencv2/opencv.hpp>
#include <map>

using json = nlohmann::json;
using stype = float;
static const auto npy_type = NPY_FLOAT32;

template<typename T> using vec = std::vector<T>;

struct ListmodeData;

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

    // trim outermost blocks in transverse direction
    static const int dim_r = ncrystals_per_ring - (2*ncrystals);

    static constexpr double energy_window = 0.2;

    static inline int ring(int blk, int xtal)
    {
        int row = ncrystals - (xtal / ncrystals) - 1;
        int blk_ax = blk % nblocks_axial;
        return row + (ncrystals + ncrystals_axial_gap)*blk_ax;
    }

    static inline int idx(int blk, int xtal)
    {
        int col = xtal % ncrystals;
        int mod = blk >> 2;
        return col + (ncrystals + ncrystals_transverse_gap)*mod;
    }
};

class CrystalLookupTable
{
    vec<vec<cv::Mat>> scaled_luts;

    public:
    static const int lut_dim = 512;
    bool loaded = false;

    CrystalLookupTable(std::string, const vec<double>&);

    inline int operator() (size_t blk, size_t scale_idx, size_t y, size_t x) const
    { return scaled_luts[blk][scale_idx].at<int>(y,x); }
};

class PhotopeakLookupTable
{
    vec<vec<double>> photopeaks;
    vec<vec<vec<double>>> doi;

    public:
    const double energy_window = Geometry::energy_window;
    bool loaded = false;

    PhotopeakLookupTable(std::string);
    static std::string find_cfg_file(std::string);

    int in_window(size_t blk, size_t xtal, double e) const
    {
        double th = photopeaks[blk][xtal];
        if (th < 0) return 0;

        double lld = (1.0 - energy_window)*th, uld = (1.0 + energy_window)*th;
        if (e < lld || e > uld) return -1;

        return (e - lld) / (uld - lld) * 63.0;
    }

    int doi_window(size_t blk, size_t xtal, double val) const
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

    vec<stype> s;
    Sinogram(int dt):
        s(vec<stype> (dt*dim_r, 0)) {};
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

        // trim outermost blocks from transverse dimension of sinogram
        r -= ncrystals;
        if (r < 0 || r > dim_r) return;

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
    vec<Sinogram> m;

    public:

    CrystalLookupTable lut;
    PhotopeakLookupTable ppeak;

    // first arg is horiz. index, second arg is vert. index
    inline Sinogram& operator() (int h, int v){ return m[v*nring + h]; };

    ListmodeData event_to_coords(const CoincidenceData&, size_t) const;
    //void write_event(std::ofstream&, const CoincidenceData&, size_t);
    void write_to(std::string);
    void read_from(std::string);

    PyObject *to_py_data();
    Michelogram(PyObject*);

    Michelogram(int dt, std::string base_dir, vec<double> scaling):
        m(vec<Sinogram> (nring*nring, Sinogram(dt))),
        lut(base_dir, scaling),
        ppeak(base_dir) {};

    std::streampos sort_span(
            std::string, std::streampos, std::streampos,
            bool, bool);

    FILE *encode_span (std::string, std::streampos, std::streampos, int) const;

    bool loaded() { return lut.loaded && ppeak.loaded; }

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
