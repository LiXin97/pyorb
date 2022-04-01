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

namespace py = pybind11;

#define kdim 2

py::tuple extract(
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

    // Allocate the arrays.
    const int num_keypoints = keypts.size();
    // Keypoints.
    py::array_t<float> pykeypoints(
            py::detail::any_container<ssize_t>(
                    {num_keypoints, kdim}
                    )
    );
    py::buffer_info pykeypoints_buf = pykeypoints.request();
    float *pykeypoints_ptr = (float *)pykeypoints_buf.ptr;
    // Scores.
    py::array_t<float> pyscores(
            py::detail::any_container<ssize_t>(
                    {num_keypoints}
                    )
    );
    py::buffer_info pyscores_buf = pyscores.request();
    float *pyscores_ptr = (float *)pyscores_buf.ptr;

    // Copy.
    for (int i = 0; i < num_keypoints; ++i) {
        float pt[2] = {keypts[i].pt.x, keypts[i].pt.y};
        memcpy(pykeypoints_ptr + kdim * i, pt, kdim * sizeof(float));
        pyscores_ptr[i] = keypts[i].response;
    }

    return py::make_tuple(pykeypoints, pyscores);
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

py::tuple extract_fromimg(
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

    // Allocate the arrays.
    const int num_keypoints = keypts.size();
    // Keypoints.
    py::array_t<float> pykeypoints(
            py::detail::any_container<ssize_t>(
                    {num_keypoints, kdim}
                    )
    );
    py::buffer_info pykeypoints_buf = pykeypoints.request();
    float *pykeypoints_ptr = (float *)pykeypoints_buf.ptr;
    // Scores.
    py::array_t<float> pyscores(
            py::detail::any_container<ssize_t>(
                    {num_keypoints}
                    )
    );
    py::buffer_info pyscores_buf = pyscores.request();
    float *pyscores_ptr = (float *)pyscores_buf.ptr;

    // Copy.
    for (int i = 0; i < num_keypoints; ++i) {
        float pt[2] = {keypts[i].pt.x, keypts[i].pt.y};
        memcpy(pykeypoints_ptr + kdim * i, pt, kdim * sizeof(float));
        pyscores_ptr[i] = keypts[i].response;
    }

    return py::make_tuple(pykeypoints, pyscores);
}