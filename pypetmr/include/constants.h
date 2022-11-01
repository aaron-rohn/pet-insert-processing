
#ifndef _CONSTANTS_H
#define _CONSTANTS_H

namespace Geometry
{
    static const int nmodules = 16;
    static const int ncrystals = 19;
    static const int nblocks = 64;
    static const int nblocks_axial = 4;
    static const int ncrystals_total = ncrystals * ncrystals;

    static const int ncrystals_axial_gap = 1;
    static const int nring = nblocks_axial * (ncrystals + ncrystals_axial_gap);

    static const int ncrystals_transverse_gap = 0;
    static const int ncrystals_per_ring = nmodules * (ncrystals + ncrystals_transverse_gap);

    static const int dim_theta_full = ncrystals_per_ring;
    static const int dim_theta_half = ncrystals_per_ring/2;

    // trim outermost blocks in transverse direction
    static const int dim_r = ncrystals_per_ring - (2*ncrystals);

    // number of DOI thresholds
    static const int ndoi = 3;

    static constexpr double energy_window = 0.2;

};

#endif
