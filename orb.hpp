//
// Created by xin on 31/3/22.
//

#ifndef ORB_ORB_HPP
#define ORB_ORB_HPP

#include <vector>
#include <list>
#include <array>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "orb_point_pairs.h"

class orb_extractor_node {
public:
    //! Constructor
    orb_extractor_node() = default;

    //! Divide node to four child nodes
    std::array<orb_extractor_node, 4> divide_node();

    //! Keypoints which distributed into this node
    std::vector<cv::KeyPoint> keypts_;

    //! Begin and end of the allocated area on the image
    cv::Point2i pt_begin_, pt_end_;

    //! A iterator pointing to self, used for removal on list
    std::list<orb_extractor_node>::iterator iter_;

    //! A flag designating if this node is a leaf node
    bool is_leaf_node_ = false;
};

class extractor {
public:
    //! Destructor
    virtual ~extractor() = default;

    //! Extract keypoints and each descriptor of them
    virtual void extract(const cv::_InputArray &in_image, const cv::_InputArray &in_image_mask,
                         std::vector<cv::KeyPoint> &keypts, const cv::_OutputArray &out_descriptors) = 0;

    virtual void initialize() = 0;

    //! Get the maximum number of keypoints
    unsigned int get_max_num_keypoints() const;

    //! Set the maximum number of keypoints
    void set_max_num_keypoints(const unsigned int max_num_keypts);

    //! Get the scale factor
    float get_scale_factor() const;

    //! Set the scale factor
    void set_scale_factor(const float scale_factor);

    //! Get the number of scale levels
    unsigned int get_num_scale_levels() const;

    //! Set the number of scale levels
    void set_num_scale_levels(const unsigned int num_levels);

    //! Get the initial fast threshold
    unsigned int get_initial_fast_threshold() const;

    //! Set the initial fast threshold
    void set_initial_fast_threshold(const unsigned int initial_fast_threshold);

    //! Get the minimum fast threshold
    unsigned int get_minimum_fast_threshold() const;

    //! Set the minimum fast threshold
    void set_minimum_fast_threshold(const unsigned int minimum_fast_threshold);

    //! Get scale factors
    std::vector<float> get_scale_factors() const;

    //! Set scale factors
    std::vector<float> get_inv_scale_factors() const;

    //! Get sigma square for all levels
    std::vector<float> get_level_sigma_sq() const;

    //! Get inverted sigma square for all levels
    std::vector<float> get_inv_level_sigma_sq() const;

protected:

    //! Number of feature points to be extracted
    unsigned int max_num_keypts_;

    //! A list of the scale factor of each pyramid layer
    std::vector<float> scale_factors_;
    std::vector<float> inv_scale_factors_;
    //! A list of the sigma of each pyramid layer
    std::vector<float> level_sigma_sq_;
    std::vector<float> inv_level_sigma_sq_;
};

class orb_extractor : public extractor {
public:


    //    //! Constructor
    //    orb_extractor(const unsigned int max_num_keypts,
    //                  const float scale_factor, const unsigned int num_levels,
    //                  const unsigned int ini_fast_thr, const unsigned int min_fast_thr,
    //                  const std::vector<std::vector<float>>& mask_rects = {});

    //! Constructor
    orb_extractor(const unsigned int max_num_keypts = 500,
                  const unsigned int num_levels = 8,
                  const float scale_factor = 1.2);

    //! Destructor
    virtual ~orb_extractor() = default;

    //! Extract keypoints and each descriptor of them
    void
    extract(const cv::_InputArray &in_image, const cv::_InputArray &in_image_mask, std::vector<cv::KeyPoint> &keypts,
            const cv::_OutputArray &out_descriptors);

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;

private:
    //! Initialize orb extractor
    void initialize();

    //! Calculate scale factors and sigmas
    void calc_scale_factors();

    //! Create a mask matrix that constructed by rectangles
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! Compute image pyramid
    void compute_image_pyramid(const cv::Mat &image);

    //! Compute fast keypoints for cells in each image pyramid
    void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>> &all_keypts, const cv::Mat &mask) const;

    //! Pick computed keypoints on the image uniformly
    std::vector<cv::KeyPoint> distribute_keypoints_via_tree(const std::vector<cv::KeyPoint> &keypts_to_distribute,
                                                            const int min_x, const int max_x, const int min_y,
                                                            const int max_y, const unsigned int num_keypts) const;

    //! Initialize nodes that used for keypoint distribution tree
    std::list<orb_extractor_node>
    initialize_nodes(const std::vector<cv::KeyPoint> &keypts_to_distribute, const int min_x,
                     const int max_x, const int min_y, const int max_y) const;

    //! Assign child nodes to the all node list
    void assign_child_nodes(const std::array<orb_extractor_node, 4> &child_nodes, std::list<orb_extractor_node> &nodes,
                            std::vector<std::pair<int, orb_extractor_node *>> &leaf_nodes) const;

    //! Find keypoint which has maximum value of response
    std::vector<cv::KeyPoint> find_keypoints_with_max_response(std::list<orb_extractor_node> &nodes) const;

    //! Compute orientation for each keypoint
    void compute_orientation(const cv::Mat &image, std::vector<cv::KeyPoint> &keypts) const;

    //! Correct keypoint's position to comply with the scale
    void correct_keypoint_scale(std::vector<cv::KeyPoint> &keypts_at_level, const unsigned int level) const;

    //! Compute the gradient direction of pixel intensity in a circle around the point
    float ic_angle(const cv::Mat &image, const cv::Point2f &point) const;

    //! Compute orb descriptors for all keypoint
    void compute_orb_descriptors(const cv::Mat &image, const std::vector<cv::KeyPoint> &keypts,
                                 cv::Mat &descriptors) const;

    //! Compute orb descriptor of a keypoint
    void compute_orb_descriptor(const cv::KeyPoint &keypt, const cv::Mat &image, uchar *desc) const;

    //! BRIEF orientation
    static constexpr unsigned int fast_patch_size_ = 31;
    //! half size of FAST patch
    static constexpr int fast_half_patch_size_ = fast_patch_size_ / 2;

    //! size of maximum ORB patch radius
    static constexpr unsigned int orb_patch_radius_ = 19;

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    //! Maximum number of keypoint of each level
    std::vector<unsigned int> num_keypts_per_level_;
    //! Index limitation that used for calculating of keypoint orientation
    std::vector<int> u_max_;

    unsigned int num_levels_;
    float scale_factor_;

    unsigned int ini_fast_thr_ = 20;
    unsigned int min_fast_thr = 7;


    std::vector<std::vector<float>> mask_rects_;
};

#endif //ORB_ORB_HPP
