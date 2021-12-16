
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <sinogram.h>
#include <singles.h>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

void Sinogram::add_event(int idx1, int idx2)
{
    int theta = (idx1 + idx2) % npix;
    int r = std::abs(idx1 - idx2);
    if (idx1 + idx2 >= npix) r = npix - r;

    std::lock_guard<std::mutex> lck(m);
    (*this)(theta, r/2)++;
}

void Michelogram::load_luts(std::string dir)
{
    if (dir == std::string())
    {
        std::cout << "Skipping LUT loading" << std::endl;
        return;
    }

    for (auto &p : std::filesystem::recursive_directory_iterator(dir))
    {
        if (p.path().extension() == ".lut")
        {
            std::string curr_path = p.path().filename();
            size_t blk = 0;
            int n = std::sscanf(curr_path.c_str(), "block%zul.lut", &blk);

            if (n != 1 || blk >= Single::nblocks)
            {
                std::cout << "Error: " << curr_path << std::endl;
                throw std::invalid_argument(curr_path);
            }

            auto &l = luts[blk];
            std::ifstream f(p.path(), std::ios::in | std::ios::binary);
            f.read((char*)l.data(), l.size()*sizeof(int));
        }
    }
}

std::string Michelogram::find_cfg_file(std::string base_dir)
{
    for (auto &p : std::filesystem::recursive_directory_iterator(base_dir))
    {
        if (p.path().extension() == ".json")
        {
            return p.path();
        }
    }
    return std::string();
}

void Michelogram::load_photopeaks(std::string cfg_file)
{
    if (cfg_file == std::string())
    {
        std::cout << "Skipping photopeak loading" << std::endl;
        return;
    }

    json cfg;
    std::ifstream(cfg_file) >> cfg;

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
            if (elem == "block" || elem == "photopeak")
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

void Michelogram::add_event(const CoincidenceData &c, bool flip_ring_num)
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
    if (flip_ring_num)
    {
        rowa = (ncrystals_per_block-1) - rowa;
        rowb = (ncrystals_per_block-1) - rowb;
    }

    int blka_ax = ba % nblocks_axial, blkb_ax = bb % nblocks_axial;
    int ra = rowa + blka_ax*ncrystals_per_block + blka_ax;
    int rb = rowb + blkb_ax*ncrystals_per_block + blkb_ax;

    auto &s = (*this)(ra, rb);

    // Calculate index along the ring

    // crystals are indexed in the opposite direction from increasing module number
    int cola = (ncrystals_per_block - 1) - (xa % ncrystals_per_block);
    int colb = (ncrystals_per_block - 1) - (xb % ncrystals_per_block);

    int moda = ba >> 2, modb = bb >> 2;
    int idx1 = cola + (ncrystals_per_block * moda);
    int idx2 = colb + (ncrystals_per_block * modb);

    s.add_event(idx1, idx2);
}

void Michelogram::write_to(std::string fname)
{
    std::ofstream f(fname, std::ios::out | std::ios::binary);
    for (auto &s : *this)
        s.write_to(f);
}

void Michelogram::read_from(std::string fname)
{
    std::ifstream f(fname, std::ios::in | std::ios::binary);
    for (auto &s : *this)
        s.read_from(f);
}

Michelogram::Michelogram(PyObject *arr):
    Michelogram()
{
    int ndim = PyArray_NDIM((PyArrayObject*)arr);
    if (ndim != 4)
    {
        std::cerr << "Invalid number of dimensions in numpy array: " <<
            ndim << std::endl;
        return;
    }

    npy_intp *dims = PyArray_DIMS((PyArrayObject*)arr);
    if (dims[0] != nring ||
        dims[1] != nring ||
        dims[2] != dim_theta ||
        dims[3] != dim_r)
    {
        std::cerr << "Invalid sinogram dimensions: ";
        for (int i = 0; i < 4; i++) std::cerr << dims[i] << " ";
        std::cerr << std::endl;
        return;
    }

    for (auto b = begin(), e = end(); b != e; ++b)
    {
        int *py_sino_ptr = (int*)PyArray_GETPTR4((PyArrayObject*)arr, b.v, b.h, 0, 0);
        std::memcpy(b->s.data(), py_sino_ptr, b->s.size()*sizeof(int));
    }
}

PyObject *Michelogram::to_py_data()
{
    npy_intp dims[] = {nring, nring, dim_theta, dim_r};
    PyObject *arr = PyArray_SimpleNew(4, dims, NPY_INT);

    for (auto b = begin(), e = end(); b != e; ++b)
    {
        int *py_sino_ptr = (int*)PyArray_GETPTR4((PyArrayObject*)arr, b.v, b.h, 0, 0);
        std::memcpy(py_sino_ptr, b->s.data(), b->s.size()*sizeof(int));
    }

    return arr;
}

std::streampos sort_sinogram_span(
        std::string fname,
        std::streampos start,
        std::streampos end,
        int flip_flood_y_coord,
        Michelogram &m,
        std::atomic_bool &stop
) {
    CoincidenceData c;
    std::ifstream f(fname, std::ios::binary);
    f.seekg(start);

    while (!stop && f.tellg() < end)
    {
        f.read((char*)&c, sizeof(c));
        m.add_event(c, flip_flood_y_coord);
    }

    return f.tellg();
};
