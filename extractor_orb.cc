//
// Created by xin on 31/3/22.
//

#include "orb.hpp"

//#include "orb_src/orb_extractor.h"
//#include "orb_src/orb_params.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Core>

namespace py = pybind11;


py::dict extract(
        const std::string &img_path,
        const int num_points = 500
) {
    auto img = cv::imread(img_path, 0);

    assert(!img.empty());

    std::vector<cv::KeyPoint> keypts;
    cv::Mat description;

    auto extractor = new orb_extractor(num_points);

//    auto param = new openvslam::feature::orb_params();
//    auto extractor = new openvslam::feature::orb_extractor(param, num_points);
    extractor->extract(img, cv::Mat(), keypts, description);


    std::vector<Eigen::Vector3d> KeyPoints;

    std::cout << "keypts.size() = " << keypts.size() << std::endl;

    for (const auto &kpt: keypts) {
        KeyPoints.emplace_back(kpt.pt.x, kpt.pt.y, kpt.response);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["KeyPoints"] = KeyPoints;

    return success_dict;
}