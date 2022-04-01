#include <iostream>

#include <pybind11/pybind11.h>

namespace py = pybind11;


#include "extractor_orb.cc"

//python setup.py bdist_wheel --plat-name=manylinux1_x86_64
//twine upload dist/pyorbfeature-0.1.4-cp39-cp39-manylinux1_x86_64.whl --verbose


PYBIND11_MODULE(pyorb, m) {
    m.doc() = "ORB plugin";
#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    // extract.
    m.def("extract", &extract,
          py::arg("img"),
          py::arg("num_points") = 500,
          py::arg("num_levels") = 8,
          py::arg("scale_factor") = 1.2,
          "Extract Orb fatures.");


    // extract_fromimg
    m.def("extract_fromimg", &extract_fromimg,
          py::arg("img_path"),
          py::arg("num_points") = 500,
          py::arg("num_levels") = 8,
          py::arg("scale_factor") = 1.2,
          "Extract Orb fatures.");

}
