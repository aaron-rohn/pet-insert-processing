#include <sinogram.h>
#include <singles.h>

void Sinogram::add_event(int idx1, int idx2)
{
    int theta = (idx1 + idx2) % npix;
    int r = idx1 > idx2 ? idx1 - idx2 : idx2 - idx1;
    if (idx1 + idx2 >= npix) r = npix - r;
    (*this)(theta, r/2)++;
}

std::vector<std::vector<int>>
Michelogram::load_luts(std::string lut_dir)
{
    // Default value of LUTs is an invalid crystal number
    std::vector<std::vector<int>> luts (
            Single::nblocks, std::vector<int>(lut_pix*lut_pix, xtal_max));

    for (auto &p : std::filesystem::recursive_directory_iterator(lut_dir))
    {
        if (p.path().extension() == ".lut")
        {
            std::string curr_path = p.path().filename();
            size_t blk = 0;
            std::sscanf(curr_path.c_str(), "block%zul.lut", &blk);

            if (blk < Single::nblocks)
            {
                auto &l = luts[blk];
                std::ifstream f(p.path(), std::ios::in | std::ios::binary);
                f.read((char*)l.data(), l.size()*sizeof(int));
            }
        }
    }

    return luts;
}

std::vector<std::vector<double>>
Michelogram::load_photopeaks(std::string base_dir)
{
    std::vector<std::vector<double>> photopeaks (
            Single::nblocks, std::vector<double>(xtal_max, -1));

    for (auto &p : std::filesystem::recursive_directory_iterator(base_dir))
    {
        if (p.path().extension() == ".json")
        {
            json cfg;
            std::ifstream(p.path()) >> cfg;

            // iterate over each block in the json file
            for (auto &[blk, values]: cfg.items())
            {
                size_t blk_num = std::stoi(blk);

                if (blk_num >= Single::nblocks)
                    continue;

                double blk_ppeak = -1;

                // iterate over each item within the block
                for (auto &[elem, ppeak]: values.items())
                {
                    // record the block photopeak
                    if (elem.compare("block") == 0)
                    {
                        blk_ppeak = ppeak;
                    }
                    else
                    {
                        // record the crystal photopeak
                        size_t xtal_num = std::stoi(elem);
                        if (xtal_num < xtal_max) photopeaks[blk_num][xtal_num] = ppeak;
                    }
                }

                // If found, assign the block photopeak to any missed crystals
                if (blk_ppeak > 0)
                {
                    for (auto &xtal_ppeak : photopeaks[blk_num])
                    {
                        if (xtal_ppeak < 0) xtal_ppeak = blk_ppeak;
                    }
                }
            }
        }
    }

    return photopeaks;
}

void Michelogram::add_event(const CoincidenceData &c)
{
    auto [ba, bb] = c.blk();
    auto [pos_xa, pos_ya, pos_xb, pos_yb] = c.pos();

    // Lookup crystal index

    int xa = luts[ba][pos_ya*lut_pix + pos_xa];
    int xb = luts[bb][pos_yb*lut_pix + pos_xb];

    if (xa >= xtal_max || xb >= xtal_max) return;

    // Apply energy thresholds

    double tha = photopeaks[ba][xa];
    double thb = photopeaks[bb][xb];

    auto [ea, eb] = c.e_sum();
    if (tha > 0 && (ea < (1.0 - energy_window)*tha ||
                    ea > (1.0 + energy_window)*tha)) return;

    if (thb > 0 && (eb < (1.0 - energy_window)*thb ||
                    eb > (1.0 + energy_window)*thb)) return;

    // Calculate ring pairs

    int rowa = xa / ncrystals_per_block;
    int rowb = xb / ncrystals_per_block;
    int ra = rowa + (ba % nblocks_axial)*ncrystals_per_block;
    int rb = rowb + (bb % nblocks_axial)*ncrystals_per_block;

    auto &s = (*this)(ra, rb);

    // Calculate index along the ring

    int moda = ba >> 2, modb = bb >> 2;

    int cola = (ncrystals_per_block - 1) - xa % ncrystals_per_block;
    int colb = (ncrystals_per_block - 1) - xb % ncrystals_per_block;

    int idx1 = cola + (ncrystals_per_block * moda);
    int idx2 = colb + (ncrystals_per_block * modb);

    s.add_event(idx1, idx2);
}

void Michelogram::write_to(std::string fname)
{
    std::ofstream f(fname, std::ios::out | std::ios::binary);

    for (int rd = 0; rd < nring; rd++)
    {
        for (auto &upper: {false, true})
        {
            for (int r0 = rd; r0 < nring; r0++)
            {
                int r1 = r0 - rd;
                auto &s = upper ? (*this)(r1, r0) : (*this)(r0, r1);
                s.write_to(f);
            }

            // no upper segment for segment 0
            if (rd == 0) break;
        }
    }
}
