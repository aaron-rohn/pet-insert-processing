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
};

class Michelogram: Geometry
{
    std::vector<std::vector<int>> luts;
    std::vector<std::vector<double>> photopeaks;
    std::vector<Sinogram> m;

    public:
    Michelogram(std::string base_dir):
        luts(load_luts(base_dir)),
        photopeaks(load_photopeaks(base_dir)),
        m(std::vector<Sinogram>(nring*nring)) {}

    inline Sinogram& operator() (int r0, int r1){ return m[r1*nring + r0]; };

    static std::vector<std::vector<double>> load_photopeaks(std::string);
    static std::vector<std::vector<int>> load_luts(std::string);

    void add_event(const CoincidenceData&);
    void write_to(std::string);
};

#endif
