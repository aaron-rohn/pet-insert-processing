
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <sinogram.h>
#include <cstdio>

std::string PhotopeakLookupTable::find_cfg_file(std::string base_dir)
{
    if (base_dir == std::string()) return base_dir;

    for (auto &p : std::filesystem::recursive_directory_iterator(base_dir))
        if (p.path().extension() == ".json") return p.path();

    return std::string();
}

PhotopeakLookupTable::PhotopeakLookupTable(std::string base_dir):
        photopeaks(vec<vec<double>> (
                    Single::nblocks,
                    vec<double>(Geometry::ncrystals_total, -1))),

        doi(vec<vec<vec<double>>> (
                    Single::nblocks,
                    vec<vec<double>>(Geometry::ncrystals_total)))
{
    std::string cfg_file = find_cfg_file(base_dir);
    if (cfg_file == std::string()) return;

    json cfg;
    std::ifstream(cfg_file) >> cfg;

    // iterate over each block in the json file
    for (auto &[blk, blk_values]: cfg.items())
    {
        size_t blk_num = std::stoi(blk);
        if (blk_num >= Single::nblocks)
        {
            std::cerr << "Error: Invalid block number in config " << blk_num << std::endl;
            return;
        }

        double blk_ppeak = blk_values["photopeak"];

        for (auto &[crystal, xtal_values]: blk_values["crystal"].items())
        {
            size_t xtal_num = std::stoi(crystal);

            if (xtal_num > Geometry::ncrystals_total)
            {
                std::cerr << "Error: Invalid crystal number in config " << xtal_num << std::endl;
                return;
            }

            if (xtal_num == Geometry::ncrystals_total)
                continue;

            photopeaks[blk_num][xtal_num] = xtal_values["energy"]["photopeak"];
            doi[blk_num][xtal_num] = xtal_values["DOI"].get<vec<double>>();
        }

        // If found, assign the block photopeak to any missed crystals
        if (blk_ppeak > 0)
        {
            for (auto &xtal_ppeak : photopeaks[blk_num])
                if (xtal_ppeak < 0) xtal_ppeak = blk_ppeak;
        }
    }

    loaded = true;
}

ListmodeData
Michelogram::event_to_coords(const CoincidenceData& c, size_t scale_idx) const
{
    // Lookup crystal index
    auto [ba, bb] = c.blk();
    auto [pos_xa, pos_ya, pos_xb, pos_yb] = c.pos();

    unsigned int xa = lut_lookup(ba, scale_idx, pos_ya, pos_xa);
    unsigned int xb = lut_lookup(bb, scale_idx, pos_yb, pos_xb);
    if (xa >= ncrystals_total || xb >= ncrystals_total)
        return ListmodeData();

    // Apply energy thresholds
    auto [ea, eb] = c.e_sum();
    int scaled_ea = ppeak.in_window(ba, xa, ea);
    int scaled_eb = ppeak.in_window(bb, xb, eb);
    if (scaled_ea < 0 || scaled_eb < 0)
        return ListmodeData();

    auto [doia_val, doib_val] = c.doi();
    unsigned int doia = ppeak.doi_window(ba, xa, doia_val);
    unsigned int doib = ppeak.doi_window(bb, xb, doib_val);

    unsigned int ra = ring(ba, xa), rb = ring(bb, xb);
    unsigned int idxa = idx(ba, xa), idxb = idx(bb, xb);

    return ListmodeData {
        .ring_a = ra, .crystal_a = idxa,
        .ring_b = rb, .crystal_b = idxb,
        .energy_b = (uint16_t)scaled_eb,
        .energy_a = (uint16_t)scaled_ea,
        .doi_b = doib, .doi_a = doia,
        .abstime = c.abstime(),
        .tdiff = c.tdiff(),
        .prompt = c.prompt()
    };
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
    m = vec<Sinogram> (nring*nring, Sinogram(dim_theta));

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
        std::streampos end,
        bool prompt, bool delay
) {
    ListmodeData lm;
    Reader rdr(fname, start, end);

    while (rdr)
    {
        rdr.read((char*)&lm, sizeof(lm));
        if ((prompt && lm.prompt) || (delay && !lm.prompt))
        {
            (*this)(lm.ring_a, lm.ring_b).add_event(
                    lm.crystal_a, lm.crystal_b);
        }
    }

    return end;
}


FILE *Michelogram::encode_span (
        std::string fname, 
        std::streampos start,
        std::streampos end,
        int scale_idx
) const {
    FILE *fout = std::tmpfile();
    Reader rdr(fname, start, end);
    CoincidenceData c;

    while (rdr)
    {
        rdr.read((char*)&c, sizeof(c));
        ListmodeData lm = event_to_coords(c, scale_idx);

        if (lm.valid())
            std::fwrite(&lm, sizeof(lm), 1, fout);
    }

    return fout;
}
