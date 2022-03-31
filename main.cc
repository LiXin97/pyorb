#include <iostream>

#include <pybind11/pybind11.h>

namespace py = pybind11;


#include "extractor_orb.cc"

PYBIND11_MODULE(pyorb, m) {
    m.doc() = "ORB plugin";
#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    // Homography Decomposition.
    m.def("extract", &extract,
          py::arg("img_path"),
          "Get Orb points.");
}
