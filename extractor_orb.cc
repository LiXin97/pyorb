//
// Created by xin on 31/3/22.
//

#include "orb.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

namespace py = pybind11;


py::dict extract(
        const std::string &img_path
) {
    auto img = cv::imread("/home/xin/Downloads/ldata/frame2849.jpg", 0);

    assert(!img.empty());

    std::vector<cv::KeyPoint> keypts;
    cv::Mat description;

    auto extractor = new orb_extractor();
    extractor->extract(img, cv::Mat(), keypts, description);


    std::vector<Eigen::Vector3d> KeyPoints;

    for (const auto &kpt: keypts) {
        KeyPoints.emplace_back(kpt.pt.x, kpt.pt.y, kpt.response);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["KeyPoints"] = KeyPoints;

    return success_dict;
}