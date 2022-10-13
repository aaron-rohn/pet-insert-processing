
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <sinogram.h>

CrystalLookupTable::CrystalLookupTable(std::string dir, std::vector<double> scaling):
        scaled_luts(vec<vec<cv::Mat>> (Single::nblocks, vec<cv::Mat>(scaling.size())))
{
    if (dir == std::string()) return;

    for (auto &p : std::filesystem::recursive_directory_iterator(dir))
    {
        if (p.path().extension() == ".lut")
        {
            std::string curr_path = p.path().filename();
            size_t blk = 0;
            int n = std::sscanf(curr_path.c_str(), "block%zul.lut", &blk);

            if (n != 1 || blk >= Single::nblocks)
            {
                std::cout << "Error: Invalid LUT filename " << curr_path << std::endl;
                return;
            }

            std::ifstream f(p.path(), std::ios::in | std::ios::binary);
            std::vector<int> l (lut_dim*lut_dim, Geometry::ncrystals_total);
            f.read((char*)l.data(), l.size()*sizeof(int));
            cv::Mat m_in(lut_dim, lut_dim, CV_32S, l.data());

            for (size_t i = 0; i < scaling.size(); i++)
            {
                cv::Mat m_out;
                cv::resize(m_in, m_out, cv::Size(), scaling[i], scaling[i], cv::INTER_NEAREST);
                int sz = (m_out.cols / 2) - (lut_dim/2);
                scaled_luts[blk][i] = cv::Mat(m_out, cv::Rect(sz, sz, lut_dim, lut_dim)).clone();
            }
        }
    }

    loaded = true;
}

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

std::tuple<bool, int, int, int, int, int, int>
Michelogram::event_to_coords(const CoincidenceData& c, size_t scale_idx)
{
    static const auto invalid_ev =
        std::make_tuple(false, -1, -1, -1, -1, -1, -1);

    // Lookup crystal index
    auto [ba, bb] = c.blk();
    auto [pos_xa, pos_ya, pos_xb, pos_yb] = c.pos();

    int xa = lut(ba, scale_idx, pos_ya, pos_xa);
    int xb = lut(bb, scale_idx, pos_yb, pos_xb);
    if (xa >= ncrystals_total || xb >= ncrystals_total)
        return invalid_ev;

    // Apply energy thresholds
    auto [ea, eb] = c.e_sum();
    if (!ppeak.in_window(ba,xa,ea) || !ppeak.in_window(bb,xb,eb))
        return invalid_ev;

    auto [doia_val, doib_val] = c.doi();
    int doia = ppeak.doi_window(ba, xa, doia_val);
    int doib = ppeak.doi_window(bb, xb, doib_val);

    int ra = ring(ba, xa), rb = ring(bb, xb);
    int idxa = idx(ba, xa), idxb = idx(bb, xb);
    return std::make_tuple(true, ra, rb, idxa, idxb, doia, doib);
}

void Michelogram::add_event(const CoincidenceData &c, size_t scale_idx)
{
    auto [valid, ra, rb, idxa, idxb, doia, doib] = event_to_coords(c, scale_idx);

    if (valid)
    {
        (*this)(ra,rb).add_event(idxa, idxb);
    }
}

void Michelogram::write_event(std::ofstream &f, const CoincidenceData &c, size_t scale_idx)
{
    auto [valid, ra, rb, idxa, idxb, doia, doib] = event_to_coords(c, scale_idx);

    if (valid)
    {
        int16_t data[] = {(int16_t)idxa, (int16_t)ra,
                          (int16_t)idxb, (int16_t)rb,
                          (int16_t)(doia << 8 | doib)}; 
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

Michelogram::Michelogram(PyObject *arr):
    lut(std::string(), vec<double>()),
    ppeak(std::string())
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
        uint64_t start, uint64_t end,
        vec<uint64_t> fpos
) {
    CoincidenceData c;
    std::ifstream f(fname, std::ios::binary);
    f.seekg(start);

    size_t idx = 0;
    uint64_t pos = start;
    size_t sz = fpos.size();

    while (f.good() && pos < end)
    {
        pos = f.tellg();
        f.read((char*)&c, sizeof(c));
        for (; idx < sz-1 && fpos[idx+1] < pos; idx++) ;
        add_event(c, idx);
    }

    return pos;
}
