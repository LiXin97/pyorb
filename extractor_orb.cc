//
// Created by xin on 31/3/22.
//

#include "orb.hpp"

//#include "orb_src/orb_extractor.h"
//#include "orb_src/orb_params.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

namespace py = pybind11;


py::dict extract(
        const py::array_t<uchar> img,
        const unsigned int num_points,
        const unsigned int num_levels,
        const float scale_factor) {

    // Check that input is grayscale.
    assert(img.ndim() == 2);

    std::vector<cv::KeyPoint> keypts;
    cv::Mat description;

    auto extractor = new orb_extractor(num_points, num_levels, scale_factor);

    py::buffer_info buf = img.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (uchar*)buf.ptr);

    //    auto param = new openvslam::feature::orb_params();
    //    auto extractor = new openvslam::feature::orb_extractor(param, num_points);
    extractor->extract(mat, cv::Mat(), keypts, description);


    std::vector<Eigen::Vector3d> KeyPoints;

    //    std::cout << "keypts.size() = " << keypts.size() << std::endl;

    for (const auto &kpt: keypts) {
        KeyPoints.emplace_back(kpt.pt.x, kpt.pt.y, kpt.response);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["KeyPoints"] = KeyPoints;

    return success_dict;

}

//py::dict extract(
//        const py::array_t<uchar> img,
//        const unsigned int num_points,
//        const unsigned int num_levels,
//        const float scale_factor) {
//
//    // Check that input is grayscale.
//    assert(img.ndim() == 2);
//
//    std::vector<cv::KeyPoint> keypts;
//    cv::Mat description;
//
//    auto extractor = new orb_extractor(num_points, num_levels, scale_factor);
//
//    py::buffer_info buf = img.request();
//    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
//
//    cv::imshow("test", mat);
//
//}

py::dict extract_fromimg(
        const std::string &img_path,
        const unsigned int num_points,
        const unsigned int num_levels,
        const float scale_factor) {
    auto img = cv::imread(img_path, 0);

    assert(!img.empty());

    std::vector<cv::KeyPoint> keypts;
    cv::Mat description;

    auto extractor = new orb_extractor(num_points, num_levels, scale_factor);

    //    auto param = new openvslam::feature::orb_params();
    //    auto extractor = new openvslam::feature::orb_extractor(param, num_points);
    extractor->extract(img, cv::Mat(), keypts, description);


    std::vector<Eigen::Vector3d> KeyPoints;

    //    std::cout << "keypts.size() = " << keypts.size() << std::endl;

    for (const auto &kpt: keypts) {
        KeyPoints.emplace_back(kpt.pt.x, kpt.pt.y, kpt.response);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["KeyPoints"] = KeyPoints;

    return success_dict;
}