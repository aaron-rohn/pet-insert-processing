
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL petmr_ARRAY_API

#include <sinogram.h>
#include <cstdio>
#include <cstring>

int Michelogram::energy_window(size_t blk, size_t xtal, size_t doi_bin, double e) const
{
    if (doi_bin >= Geometry::ndoi) return -1;

    double th = *(double*)PyArray_GETPTR3(photopeaks, blk, xtal, doi_bin);

    if (th < 0 || energy_window_width < 0) return 0;

    double lld = (1.0 - energy_window_width)*th;
    double uld = (1.0 + energy_window_width)*th;
    if (e < lld || e > uld) return -1;

    return (e - lld) / (uld - lld) * energy_scale;
}

int Michelogram::doi_window(size_t blk, size_t xtal, double val) const
{
    int i;
    double *th = (double*)PyArray_GETPTR3(doi_thresholds, blk, xtal, 0);

    for (i = 0; i < Geometry::ndoi && val < th[i]; i++)
        ;

    return i;
}

ListmodeData
Michelogram::event_to_coords(const CoincidenceData& c) const
{
    auto [ba, bb] = c.blk();

    // Lookup crystal index
    auto [pos_xa, pos_ya, pos_xb, pos_yb] = c.pos();
    size_t xa = lut_lookup(ba, pos_ya, pos_xa);
    size_t xb = lut_lookup(bb, pos_yb, pos_xb);
    if (xa >= Geometry::ncrystals_total || xb >= Geometry::ncrystals_total)
        return ListmodeData();

    // identify DOI bin
    auto [doia_val, doib_val] = c.doi();
    unsigned int doia = doi_window(ba, xa, doia_val);
    unsigned int doib = doi_window(bb, xb, doib_val);
    if (doia >= Geometry::ndoi || doib >= Geometry::ndoi)
        return ListmodeData();

    // Apply energy thresholds
    auto [ea, eb] = c.e_sum();
    int scaled_ea = energy_window(ba, xa, doia, ea);
    int scaled_eb = energy_window(bb, xb, doib, eb);
    if (scaled_ea < 0 || scaled_eb < 0)
        return ListmodeData();

    unsigned int ra = Sinogram::ring(ba, xa), rb = Sinogram::ring(bb, xb);
    unsigned int idxa = Sinogram::idx(ba, xa), idxb = Sinogram::idx(bb, xb);

    return ListmodeData {
        .ring_a = ra, .crystal_a = idxa,
        .ring_b = rb, .crystal_b = idxb,
        .energy_b = (uint16_t)scaled_eb,
        .energy_a = (uint16_t)scaled_ea,
        .doi_b = doib, .doi_a = doia,
        .abstime = c.abstime,
        .tdiff = c.tdiff,
        .prompt = c.prompt
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

Michelogram::Michelogram(PyObject *arr):
    photopeaks(NULL), doi_thresholds(NULL), lut(NULL)
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
    m = std::vector<Sinogram> (
            Geometry::nring*Geometry::nring,
            Sinogram(dim_theta));

    if (dims[0] != Geometry::nring ||
        dims[1] != Geometry::nring ||
        dims[3] != Geometry::dim_r)
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
    int dim_theta = begin()->s.size() / Geometry::dim_r;

    npy_intp dims[] = {Geometry::nring, Geometry::nring, dim_theta, Geometry::dim_r};
    PyObject *arr = PyArray_SimpleNew(4, dims, npy_type);

    for (auto b = begin(), e = end(); b != e; ++b)
    {
        stype *py_sino_ptr = (stype*)PyArray_GETPTR4((PyArrayObject*)arr, b.v, b.h, 0, 0);
        std::memcpy(py_sino_ptr, b->s.data(), b->s.size()*sizeof(stype));
    }

    return arr;
}

std::streampos Michelogram::add_to_sinogram(
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
            if (lm.doi_a <= max_doi && lm.doi_b <= max_doi)
            {
                (*this)(lm.ring_a, lm.ring_b).add_event(
                        lm.crystal_a, lm.crystal_b);
            }
        }
    }

    return end;
}

FILE *Michelogram::save_listmode (
        std::string fname, 
        std::streampos start,
        std::streampos end
) const {
    FILE *fout = std::tmpfile();
    Reader rdr(fname, start, end);
    CoincidenceData c;

    while (rdr)
    {
        rdr.read((char*)&c, sizeof(c));
        ListmodeData lm = event_to_coords(c);

        if (lm.valid())
            std::fwrite(&lm, sizeof(lm), 1, fout);
    }

    return fout;
}
