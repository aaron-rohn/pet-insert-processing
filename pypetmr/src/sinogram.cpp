
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <sinogram.h>

void CrystalLookupTable::load(std::string dir)
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

std::string PhotopeakLookupTable::find_cfg_file(std::string base_dir)
{
    for (auto &p : std::filesystem::recursive_directory_iterator(base_dir))
        if (p.path().extension() == ".json") return p.path();

    std::cout << "No photopeak file found" << std::endl;
    return std::string();
}

void PhotopeakLookupTable::load(std::string base_dir)
{
    std::string cfg_file = find_cfg_file(base_dir);

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
                if (xtal_num < Geometry::ncrystals_total)
                    photopeaks[blk_num][xtal_num] = ppeak;
            }
        }

        // If found, assign the block photopeak to any missed crystals
        if (blk_ppeak > 0)
        {
            for (auto &xtal_ppeak : photopeaks[blk_num])
                if (xtal_ppeak < 0) xtal_ppeak = blk_ppeak;
        }
    }
}

std::tuple<bool, int, int, int, int>
Michelogram::event_to_coords(const CoincidenceData& c)
{
    static const auto invalid_ev =
        std::make_tuple(false, -1, -1, -1, -1);

    // Lookup crystal index
    auto [ba, bb] = c.blk();
    auto [pos_xa, pos_ya, pos_xb, pos_yb] = c.pos();
    int xa = lut(ba, pos_ya, pos_xa), xb = lut(bb, pos_yb, pos_xb);
    if (xa >= ncrystals_total || xb >= ncrystals_total)
        return invalid_ev;

    // Apply energy thresholds
    auto [ea, eb] = c.e_sum();
    if (!ppeak.in_window(ba,xa,ea) || !ppeak.in_window(bb,xb,eb))
        return invalid_ev;

    int ra = ring(ba, xa), rb = ring(bb, xb);
    int idxa = idx(ba, xa), idxb = idx(bb, xb);
    return std::make_tuple(true, ra, rb, idxa, idxb);
}

void Michelogram::add_event(const CoincidenceData &c)
{
    auto [valid, ra, rb, idxa, idxb] = event_to_coords(c);

    if (valid)
    {
        (*this)(ra,rb).add_event(idxa, idxb);
    }
}

void Michelogram::write_event(std::ofstream &f, const CoincidenceData &c)
{
    auto [valid, ra, rb, idxa, idxb] = event_to_coords(c);

    if (valid)
    {
        int16_t data[] = {(int16_t)idxa, (int16_t)ra,
                          (int16_t)idxb, (int16_t)rb, 0}; 
        f.write((char*)data, sizeof(data));
    }
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

Michelogram::Michelogram(PyObject *arr)
{
    if (PyArray_TYPE((PyArrayObject*)arr) != npy_type)
    {
        std::cout << "Invalid type" << std::endl;
        return;
    }

    int ndim = PyArray_NDIM((PyArrayObject*)arr);
    if (ndim != 4)
    {
        std::cout << "Invalid number of dimensions in numpy array: " <<
            ndim << std::endl;
        return;
    }

    npy_intp *dims = PyArray_DIMS((PyArrayObject*)arr);
    
    int dim_theta = dims[2];
    m = std::vector<Sinogram> (nring*nring, Sinogram(dim_theta));

    if (dims[0] != nring ||
        dims[1] != nring ||
        dims[3] != dim_r)
    {
        std::cout << "Invalid sinogram dimensions: ";
        for (int i = 0; i < 4; i++) std::cout << dims[i] << " ";
        std::cout << std::endl;
        return;
    }

    for (auto b = begin(), e = end(); b != e; ++b)
    {
        stype *py_sino_ptr = (stype*)PyArray_GETPTR4((PyArrayObject*)arr, b.v, b.h, 0, 0);
        std::memcpy(b->s.data(), py_sino_ptr, b->s.size()*sizeof(stype));
    }
}

PyObject *Michelogram::to_py_data()
{
    int dim_theta = begin()->s.size() / dim_r;

    npy_intp dims[] = {nring, nring, dim_theta, dim_r};
    PyObject *arr = PyArray_SimpleNew(4, dims, npy_type);

    for (auto b = begin(), e = end(); b != e; ++b)
    {
        stype *py_sino_ptr = (stype*)PyArray_GETPTR4((PyArrayObject*)arr, b.v, b.h, 0, 0);
        std::memcpy(py_sino_ptr, b->s.data(), b->s.size()*sizeof(stype));
    }

    return arr;
}

std::streampos Michelogram::sort_span(
        std::string fname,
        std::streampos start,
        std::streampos end
) {
    CoincidenceData c;
    std::ifstream f(fname, std::ios::binary);
    f.seekg(start);

    while (f.good() && f.tellg() < end)
    {
        f.read((char*)&c, sizeof(c));
        add_event(c);
    }

    return f.tellg();
}
