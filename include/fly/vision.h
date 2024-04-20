// Copyright 2023 The EA Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#include <fly/defines.h>
#include <fly/features.h>

#ifdef __cplusplus
namespace fly
{
class array;

/**
    C++ Interface for FAST feature detector

    \param[in] in array containing a grayscale image (color images are not
               supported)
    \param[in] thr FAST threshold for which a pixel of the circle around
               the central pixel is considered to be greater or smaller
    \param[in] arc_length length of arc (or sequential segment) to be tested,
               must be within range [9-16]
    \param[in] non_max performs non-maximal suppression if true
    \param[in] feature_ratio maximum ratio of features to detect, the maximum
               number of features is calculated by feature_ratio * in.elements().
               The maximum number of features is not based on the score, instead,
               features detected after the limit is reached are discarded
    \param[in] edge is the length of the edges in the image to be discarded
               by FAST (minimum is 3, as the radius of the circle)
    \return    features object containing arrays for x and y coordinates and
               score, while array orientation is set to 0 as FAST does not
               compute orientation, and size is set to 1 as FAST does not
               compute multiple scales

    \ingroup cv_func_fast
 */
FLY_API features fast(const array& in, const float thr=20.0f, const unsigned arc_length=9,
                    const bool non_max=true, const float feature_ratio=0.05f,
                    const unsigned edge=3);

/**
    C++ Interface for Harris corner detector

    \param[in] in array containing a grayscale image (color images are not
               supported)
    \param[in] max_corners maximum number of corners to keep, only retains
               those with highest Harris responses
    \param[in] min_response minimum response in order for a corner to be
               retained, only used if max_corners = 0
    \param[in] sigma the standard deviation of a circular window (its
               dimensions will be calculated according to the standard
               deviation), the covariation matrix will be calculated to a
               circular neighborhood of this standard deviation (only used
               when block_size == 0, must be >= 0.5f and <= 5.0f)
    \param[in] block_size square window size, the covariation matrix will be
               calculated to a square neighborhood of this size (must be
               >= 3 and <= 31)
    \param[in] k_thr Harris constant, usually set empirically to 0.04f (must
               be >= 0.01f)
    \return    features object containing arrays for x and y coordinates and
               score (Harris response), while arrays orientation and size are
               set to 0 and 1, respectively, because Harris does not compute
               that information

    \ingroup cv_func_harris
 */
FLY_API features harris(const array& in, const unsigned max_corners=500,
                      const float min_response=1e5f, const float sigma=1.f,
                      const unsigned block_size=0, const float k_thr=0.04f);

/**
    C++ Interface for ORB feature descriptor

    \param[out] feat features object composed of arrays for x and y
                coordinates, score, orientation and size of selected features
    \param[out] desc Nx8 array containing extracted descriptors, where N is the
                number of selected features
    \param[in]  image array containing a grayscale image (color images are not
                supported)
    \param[in]  fast_thr FAST threshold for which a pixel of the circle around
                the central pixel is considered to be brighter or darker
    \param[in]  max_feat maximum number of features to hold (will only keep the
                max_feat features with higher Harris responses)
    \param[in]  scl_fctr factor to downsample the input image, meaning that
                each level will hold prior level dimensions divided by scl_fctr
    \param[in]  levels number of levels to be computed for the image pyramid
    \param[in]  blur_img blur image with a Gaussian filter with sigma=2 before
                computing descriptors to increase robustness against noise if
                true

    \ingroup cv_func_orb
 */
FLY_API void orb(features& feat, array& desc, const array& image,
               const float fast_thr=20.f, const unsigned max_feat=400,
               const float scl_fctr=1.5f, const unsigned levels=4,
               const bool blur_img=false);

/**
    C++ Interface for SIFT feature detector and descriptor

    \param[out] feat features object composed of arrays for x and y
                coordinates, score, orientation and size of selected features
    \param[out] desc Nx128 array containing extracted descriptors, where N is the
                number of features found by SIFT
    \param[in]  in array containing a grayscale image (color images are not
                supported)
    \param[in]  n_layers number of layers per octave, the number of octaves is
                computed automatically according to the input image dimensions,
                the original SIFT paper suggests 3
    \param[in]  contrast_thr threshold used to filter out features that have
                low contrast, the original SIFT paper suggests 0.04
    \param[in]  edge_thr threshold used to filter out features that are too
                edge-like, the original SIFT paper suggests 10.0
    \param[in]  init_sigma the sigma value used to filter the input image at
                the first octave, the original SIFT paper suggests 1.6
    \param[in]  double_input if true, the input image dimensions will be
                doubled and the doubled image will be used for the first octave
    \param[in]  intensity_scale the inverse of the difference between the minimum
                and maximum grayscale intensity value, e.g.: if the ranges are
                0-256, the proper intensity_scale value is 1/256, if the ranges
                are 0-1, the proper intensity-scale value is 1/1
    \param[in]  feature_ratio maximum ratio of features to detect, the maximum
                number of features is calculated by feature_ratio * in.elements().
                The maximum number of features is not based on the score, instead,
                features detected after the limit is reached are discarded

    \ingroup cv_func_sift
 */
FLY_API void sift(features& feat, array& desc, const array& in, const unsigned n_layers=3,
                const float contrast_thr=0.04f, const float edge_thr=10.f,
                const float init_sigma=1.6f, const bool double_input=true,
                const float intensity_scale=0.00390625f, const float feature_ratio=0.05f);

/**
    C++ Interface for SIFT feature detector and GLOH descriptor

    \param[out] feat features object composed of arrays for x and y
                coordinates, score, orientation and size of selected features
    \param[out] desc Nx272 array containing extracted GLOH descriptors, where N
                is the number of features found by SIFT
    \param[in]  in array containing a grayscale image (color images are not
                supported)
    \param[in]  n_layers number of layers per octave, the number of octaves is
                computed automatically according to the input image dimensions,
                the original SIFT paper suggests 3
    \param[in]  contrast_thr threshold used to filter out features that have
                low contrast, the original SIFT paper suggests 0.04
    \param[in]  edge_thr threshold used to filter out features that are too
                edge-like, the original SIFT paper suggests 10.0
    \param[in]  init_sigma the sigma value used to filter the input image at
                the first octave, the original SIFT paper suggests 1.6
    \param[in]  double_input if true, the input image dimensions will be
                doubled and the doubled image will be used for the first octave
    \param[in]  intensity_scale the inverse of the difference between the minimum
                and maximum grayscale intensity value, e.g.: if the ranges are
                0-256, the proper intensity_scale value is 1/256, if the ranges
                are 0-1, the proper intensity-scale value is 1/1
    \param[in]  feature_ratio maximum ratio of features to detect, the maximum
                number of features is calculated by feature_ratio * in.elements().
                The maximum number of features is not based on the score, instead,
                features detected after the limit is reached are discarded

    \ingroup cv_func_sift
 */
FLY_API void gloh(features& feat, array& desc, const array& in, const unsigned n_layers=3,
                const float contrast_thr=0.04f, const float edge_thr=10.f,
                const float init_sigma=1.6f, const bool double_input=true,
                const float intensity_scale=0.00390625f, const float feature_ratio=0.05f);

/**
   C++ Interface wrapper for Hamming matcher

   \param[out] idx is an array of MxN size, where M is equal to the number of query
               features and N is equal to n_dist. The value at position IxJ indicates
               the index of the Jth smallest distance to the Ith query value in the
               train data array.
               the index of the Ith smallest distance of the Mth query.
   \param[out] dist is an array of MxN size, where M is equal to the number of query
               features and N is equal to n_dist. The value at position IxJ indicates
               the Hamming distance of the Jth smallest distance to the Ith query
               value in the train data array.
   \param[in]  query is the array containing the data to be queried
   \param[in]  train is the array containing the data used as training data
   \param[in]  dist_dim indicates the dimension to analyze for distance (the dimension
               indicated here must be of equal length for both query and train arrays)
   \param[in]  n_dist is the number of smallest distances to return (currently, only
               values <= 256 are supported)

   \note Note: This is a special case of the \ref nearestNeighbour function with FLY_SHD
    as dist_type

   \ingroup cv_func_hamming_matcher
 */
FLY_API void hammingMatcher(array& idx, array& dist,
                          const array& query, const array& train,
                          const dim_t dist_dim=0, const unsigned n_dist=1);

/**
   C++ interface wrapper for determining the nearest neighbouring points to a
   given set of points

   \param[out] idx       is an array of \f$M \times N\f$ size, where \f$M\f$ is
                         \p n_dist and \f$N\f$ is the number of queries. The
                         value at position \f$i,j\f$ is the index of the point
                         in \p train along dim1 (if \p dist_dim is 0) or along
                         dim 0 (if \p dist_dim is 1), with the \f$ith\f$
                         smallest distance to the \f$jth\f$ \p query point.
   \param[out] dist      is an array of \f$M \times N\f$ size, where \f$M\f$ is
                         \p n_dist and \f$N\f$ is the number of queries. The
                         value at position \f$i,j\f$ is the distance from the
                         \f$jth\f$ query point to the point in \p train referred
                         to by \p idx(\f$i,j\f$). This distance is computed
                         according to the \p dist_type chosen.
   \param[in]  query     is the array containing the points to be queried. The
                         points must be described along dim0 and listed along
                         dim1 if \p dist_dim is 0, or vice versa if \p dist_dim
                         is 1.
   \param[in]  train     is the array containing the points used as training
                         data. The points must be described along dim0 and
                         listed along dim1 if \p dist_dim is 0, or vice versa if
                         \p dist_dim is 1.
   \param[in]  dist_dim  indicates the dimension that the distance computation
                         will use to determine a point's coordinates. The \p
                         train and \p query arrays must both use this dimension
                         for describing a point's coordinates
   \param[in]  n_dist    is the number of nearest neighbour points to return
                         (currently only values <= 256 are supported)
   \param[in]  dist_type is the distance computation type. Currently \ref
                         FLY_SAD (sum of absolute differences), \ref FLY_SSD (sum
                         of squared differences), and \ref FLY_SHD (hamming
                         distances) are supported.

   \ingroup cv_func_nearest_neighbour
 */
FLY_API void nearestNeighbour(array& idx, array& dist,
                            const array& query, const array& train,
                            const dim_t dist_dim=0, const unsigned n_dist=1,
                            const fly_match_type dist_type = FLY_SSD);

/**
   C++ Interface for image template matching

   \param[in]  searchImg is an array with image data
   \param[in]  templateImg is the template we are looking for in the image
   \param[in]  mType is metric that should be used to calculate the disparity
               between window in the image and the template image. It can be one of
               the values defined by the enum \ref fly_match_type
   \return     array with disparity values for the window starting at
               corresponding pixel position

   \note If \p search_img is 3d array, a batch operation will be performed.

   \ingroup cv_func_match_template
 */
FLY_API array matchTemplate(const array &searchImg, const array &templateImg, const matchType mType=FLY_SAD);

/**
   C++ Interface for SUSAN corner detector

   \param[in]  in is input grayscale/intensity image
   \param[in]  radius nucleus radius for each pixel neighborhood
   \param[in]  diff_thr intensity difference threshold
   \param[in]  geom_thr geometric threshold a.k.a **t** from equations in description
   \param[in]  feature_ratio is maximum number of features that will be returned by the function
   \param[in]  edge indicates how many pixels width area should be skipped for corner detection
   \return If SUSAN corner detection is successfull returns an object of Features class, composed of arrays for x and y
               coordinates, score, orientation and size of selected features, otherwise exception is thrown.

   \note If \p in is a 3d array, a batch operation will be performed.

   \ingroup cv_func_susan
*/
FLY_API features susan(const array& in,
                     const unsigned radius=3,
                     const float diff_thr=32.0f,
                     const float geom_thr=10.0f,
                     const float feature_ratio=0.05f,
                     const unsigned edge=3);

/**
   C++ Interface wrapper for Difference of Gaussians

   \param[in] in is input image
   \param[in] radius1 is the radius of first gaussian kernel
   \param[in] radius2 is the radius of second gaussian kernel
   \return    Difference of smoothed inputs

   \ingroup cv_func_dog
 */
FLY_API array dog(const array& in, const int radius1, const int radius2);

/**
   C++ Interface for Homography estimation

   \param[out] H is a 3x3 array containing the estimated homography.
   \param[out] inliers is the number of inliers that the homography was estimated to comprise,
               in the case that htype is FLY_HOMOGRAPHY_RANSAC, a higher inlier_thr value will increase the
               estimated inliers. Note that if the number of inliers is too low, it is likely
               that a bad homography will be returned.
   \param[in]  x_src x coordinates of the source points.
   \param[in]  y_src y coordinates of the source points.
   \param[in]  x_dst x coordinates of the destination points.
   \param[in]  y_dst y coordinates of the destination points.
   \param[in]  htype can be FLY_HOMOGRAPHY_RANSAC, for which a RANdom SAmple Consensus will be
               used to evaluate the homography quality (e.g., number of inliers), or FLY_HOMOGRAPHY_LMEDS,
               which will use Least Median of Squares method to evaluate homography quality
   \param[in]  inlier_thr if htype is FLY_HOMOGRAPHY_RANSAC, this parameter will five the maximum L2-distance
               for a point to be considered an inlier.
   \param[in]  iterations maximum number of iterations when htype is FLY_HOMOGRAPHY_RANSAC and backend is CPU,
               if backend is CUDA, iterations is the total number of iterations, an
               iteration is a selection of 4 random points for which the homography is estimated
               and evaluated for number of inliers.
   \param[in]  otype the array type for the homography output.

   \ingroup cv_func_homography
*/
FLY_API void homography(array& H, int& inliers, const array& x_src, const array& y_src,
                      const array& x_dst, const array& y_dst, const fly_homography_type htype=FLY_HOMOGRAPHY_RANSAC,
                      const float inlier_thr=3.f, const unsigned iterations=1000, const dtype otype=f32);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
        C Interface for FAST feature detector

        \param[out] out struct containing arrays for x and y
                    coordinates and score, while array orientation is set to 0
                    as FAST does not compute orientation, and size is set to 1
                    as FAST does not compute multiple scales
        \param[in]  in array containing a grayscale image (color images are
                    not supported)
        \param[in]  thr FAST threshold for which a pixel of the circle around
                    the central pixel is considered to be greater or smaller
        \param[in]  arc_length length of arc (or sequential segment) to be
                    tested, must be within range [9-16]
        \param[in]  non_max performs non-maximal suppression if true
        \param[in]  feature_ratio maximum ratio of features to detect, the
                    maximum number of features is calculated by
                    feature_ratio * in.elements(). The maximum number of
                    features is not based on the score, instead, features
                    detected after the limit is reached are discarded
        \param[in]  edge is the length of the edges in the image to be
                    discarded by FAST (minimum is 3, as the radius of the
                    circle)

        \ingroup cv_func_fast
    */
    FLY_API fly_err fly_fast(fly_features *out, const fly_array in, const float thr, const unsigned arc_length,
                         const bool non_max, const float feature_ratio, const unsigned edge);

    /**
        C Interface for Harris corner detector

        \param[out] out struct containing arrays for x and y
                    coordinates and score (Harris response), while arrays
                    orientation and size are set to 0 and 1, respectively,
                    because Harris does not compute that information
        \param[in]  in array containing a grayscale image (color images are not
                    supported)
        \param[in]  max_corners maximum number of corners to keep, only retains
                    those with highest Harris responses
        \param[in]  min_response minimum response in order for a corner to be
                    retained, only used if max_corners = 0
        \param[in]  sigma the standard deviation of a circular window (its
                    dimensions will be calculated according to the standard
                    deviation), the covariation matrix will be calculated to a
                    circular neighborhood of this standard deviation (only used
                    when block_size == 0, must be >= 0.5f and <= 5.0f)
        \param[in]  block_size square window size, the covariation matrix will be
                    calculated to a square neighborhood of this size (must be
                    >= 3 and <= 31)
        \param[in]  k_thr Harris constant, usually set empirically to 0.04f (must
                    be >= 0.01f)

        \ingroup cv_func_harris
    */
    FLY_API fly_err fly_harris(fly_features *out, const fly_array in, const unsigned max_corners,
                           const float min_response, const float sigma,
                           const unsigned block_size, const float k_thr);

    /**
        C Interface for ORB feature descriptor

        \param[out] feat fly_features struct composed of arrays for x and y
                    coordinates, score, orientation and size of selected features
        \param[out] desc Nx8 array containing extracted descriptors, where N is the
                    number of selected features
        \param[in]  in array containing a grayscale image (color images are not
                    supported)
        \param[in]  fast_thr FAST threshold for which a pixel of the circle around
                    the central pixel is considered to be brighter or darker
        \param[in]  max_feat maximum number of features to hold (will only keep the
                    max_feat features with higher Harris responses)
        \param[in]  scl_fctr factor to downsample the input image, meaning that
                    each level will hold prior level dimensions divided by scl_fctr
        \param[in]  levels number of levels to be computed for the image pyramid
        \param[in]  blur_img blur image with a Gaussian filter with sigma=2 before
                    computing descriptors to increase robustness against noise if
                    true

        \ingroup cv_func_orb
    */
    FLY_API fly_err fly_orb(fly_features *feat, fly_array *desc, const fly_array in,
                        const float fast_thr, const unsigned max_feat, const float scl_fctr,
                        const unsigned levels, const bool blur_img);

    /**
        C++ Interface for SIFT feature detector and descriptor

        \param[out] feat fly_features object composed of arrays for x and y
                    coordinates, score, orientation and size of selected features
        \param[out] desc Nx128 array containing extracted descriptors, where N is the
                    number of features found by SIFT
        \param[in]  in array containing a grayscale image (color images are not
                    supported)
        \param[in]  n_layers number of layers per octave, the number of octaves is
                    computed automatically according to the input image dimensions,
                    the original SIFT paper suggests 3
        \param[in]  contrast_thr threshold used to filter out features that have
                    low contrast, the original SIFT paper suggests 0.04
        \param[in]  edge_thr threshold used to filter out features that are too
                    edge-like, the original SIFT paper suggests 10.0
        \param[in]  init_sigma the sigma value used to filter the input image at
                    the first octave, the original SIFT paper suggests 1.6
        \param[in]  double_input if true, the input image dimensions will be
                    doubled and the doubled image will be used for the first octave
        \param[in]  intensity_scale the inverse of the difference between the minimum
                    and maximum grayscale intensity value, e.g.: if the ranges are
                    0-256, the proper intensity_scale value is 1/256, if the ranges
                    are 0-1, the proper intensity-scale value is 1/1
        \param[in]  feature_ratio maximum ratio of features to detect, the maximum
                    number of features is calculated by feature_ratio * in.elements().
                    The maximum number of features is not based on the score, instead,
                    features detected after the limit is reached are discarded

        \ingroup cv_func_sift
    */
    FLY_API fly_err fly_sift(fly_features *feat, fly_array *desc, const fly_array in,
                         const unsigned n_layers, const float contrast_thr, const float edge_thr,
                         const float init_sigma, const bool double_input,
                         const float intensity_scale, const float feature_ratio);

    /**
        C++ Interface for SIFT feature detector and GLOH descriptor

        \param[out] feat fly_features object composed of arrays for x and y
                    coordinates, score, orientation and size of selected features
        \param[out] desc Nx272 array containing extracted GLOH descriptors, where N
                    is the number of features found by SIFT
        \param[in]  in array containing a grayscale image (color images are not
                    supported)
        \param[in]  n_layers number of layers per octave, the number of octaves is
                    computed automatically according to the input image dimensions,
                    the original SIFT paper suggests 3
        \param[in]  contrast_thr threshold used to filter out features that have
                    low contrast, the original SIFT paper suggests 0.04
        \param[in]  edge_thr threshold used to filter out features that are too
                    edge-like, the original SIFT paper suggests 10.0
        \param[in]  init_sigma the sigma value used to filter the input image at
                    the first octave, the original SIFT paper suggests 1.6
        \param[in]  double_input if true, the input image dimensions will be
                    doubled and the doubled image will be used for the first octave
        \param[in]  intensity_scale the inverse of the difference between the minimum
                    and maximum grayscale intensity value, e.g.: if the ranges are
                    0-256, the proper intensity_scale value is 1/256, if the ranges
                    are 0-1, the proper intensity-scale value is 1/1
        \param[in]  feature_ratio maximum ratio of features to detect, the maximum
                    number of features is calculated by feature_ratio * in.elements().
                    The maximum number of features is not based on the score, instead,
                    features detected after the limit is reached are discarded

        \ingroup cv_func_sift
    */
    FLY_API fly_err fly_gloh(fly_features *feat, fly_array *desc, const fly_array in,
                         const unsigned n_layers, const float contrast_thr,
                         const float edge_thr, const float init_sigma, const bool double_input,
                         const float intensity_scale, const float feature_ratio);

    /**
       C Interface wrapper for Hamming matcher

       \param[out] idx is an array of MxN size, where M is equal to the number of query
                   features and N is equal to n_dist. The value at position IxJ indicates
                   the index of the Jth smallest distance to the Ith query value in the
                   train data array.
                   the index of the Ith smallest distance of the Mth query.
       \param[out] dist is an array of MxN size, where M is equal to the number of query
                   features and N is equal to n_dist. The value at position IxJ indicates
                   the Hamming distance of the Jth smallest distance to the Ith query
                   value in the train data array.
       \param[in]  query is the array containing the data to be queried
       \param[in]  train is the array containing the data used as training data
       \param[in]  dist_dim indicates the dimension to analyze for distance (the dimension
                   indicated here must be of equal length for both query and train arrays)
       \param[in]  n_dist is the number of smallest distances to return (currently, only
                   values <= 256 are supported)

       \ingroup cv_func_hamming_matcher
    */
    FLY_API fly_err fly_hamming_matcher(fly_array* idx, fly_array* dist,
                                    const fly_array query, const fly_array train,
                                    const dim_t dist_dim, const unsigned n_dist);

/**
   C++ interface wrapper for determining the nearest neighbouring points to a
   given set of points

   \param[out] idx       is an array of \f$M \times N\f$ size, where \f$M\f$ is
                         \p n_dist and \f$N\f$ is the number of queries. The
                         value at position \f$i,j\f$ is the index of the point
                         in \p train along dim1 (if \p dist_dim is 0) or along
                         dim 0 (if \p dist_dim is 1), with the \f$ith\f$
                         smallest distance to the \f$jth\f$ \p query point.
   \param[out] dist      is an array of \f$M \times N\f$ size, where \f$M\f$ is
                         \p n_dist and \f$N\f$ is the number of queries. The
                         value at position \f$i,j\f$ is the distance from the
                         \f$jth\f$ query point to the point in \p train referred
                         to by \p idx(\f$i,j\f$). This distance is computed
                         according to the \p dist_type chosen.
   \param[in]  query     is the array containing the points to be queried. The
                         points must be described along dim0 and listed along
                         dim1 if \p dist_dim is 0, or vice versa if \p dist_dim
                         is 1.
   \param[in]  train     is the array containing the points used as training
                         data. The points must be described along dim0 and
                         listed along dim1 if \p dist_dim is 0, or vice versa if
                         \p dist_dim is 1.
   \param[in]  dist_dim  indicates the dimension that the distance computation
                         will use to determine a point's coordinates. The \p
                         train and \p query arrays must both use this dimension
                         for describing a point's coordinates
   \param[in]  n_dist    is the number of nearest neighbour points to return
                         (currently only values <= 256 are supported)
   \param[in]  dist_type is the distance computation type. Currently \ref
                         FLY_SAD (sum of absolute differences), \ref FLY_SSD (sum
                         of squared differences), and \ref FLY_SHD (hamming
                         distances) are supported.

   \ingroup cv_func_nearest_neighbour
 */
FLY_API fly_err fly_nearest_neighbour(fly_array* idx, fly_array* dist,
                                  const fly_array query, const fly_array train,
                                  const dim_t dist_dim, const unsigned n_dist,
                                  const fly_match_type dist_type);

    /**
       C Interface for image template matching

       \param[out] out will have disparity values for the window starting at
                   corresponding pixel position
       \param[in]  search_img is an array with image data
       \param[in]  template_img is the template we are looking for in the image
       \param[in]  m_type is metric that should be used to calculate the disparity
                   between window in the image and the template image. It can be one of
                   the values defined by the enum \ref fly_match_type
       \return     \ref FLY_SUCCESS if disparity metric is computed successfully,
       otherwise an appropriate error code is returned.

       \note If \p search_img is 3d array, a batch operation will be performed.

       \ingroup cv_func_match_template
    */
    FLY_API fly_err fly_match_template(fly_array *out, const fly_array search_img,
                                   const fly_array template_img, const fly_match_type m_type);

    /**
       C Interface for SUSAN corner detector

       \param[out] out is fly_features struct composed of arrays for x and y
                   coordinates, score, orientation and size of selected features
       \param[in]  in is input grayscale/intensity image
       \param[in]  radius nucleus radius for each pixel neighborhood
       \param[in]  diff_thr intensity difference threshold a.k.a **t** from equations in description
       \param[in]  geom_thr geometric threshold
       \param[in]  feature_ratio is maximum number of features that will be returned by the function
       \param[in]  edge indicates how many pixels width area should be skipped for corner detection
       \return \ref FLY_SUCCESS if SUSAN corner detection is successfull, otherwise an appropriate
       error code is returned.

       \note If \p in is a 3d array, a batch operation will be performed.

       \ingroup cv_func_susan
    */
    FLY_API fly_err fly_susan(fly_features* out, const fly_array in, const unsigned radius,
                          const float diff_thr, const float geom_thr,
                          const float feature_ratio, const unsigned edge);

    /**
       C Interface wrapper for Difference of Gaussians

       \param[out] out is difference of smoothed inputs
       \param[in] in is input image
       \param[in] radius1 is the radius of first gaussian kernel
       \param[in] radius2 is the radius of second gaussian kernel
       \return    \ref FLY_SUCCESS if the computation is is successful,
                  otherwise an appropriate error code is returned.

       \ingroup cv_func_dog
     */
    FLY_API fly_err fly_dog(fly_array *out, const fly_array in, const int radius1, const int radius2);

    /**
       C Interface wrapper for Homography estimation

       \param[out] H is a 3x3 array containing the estimated homography.
       \param[out] inliers is the number of inliers that the homography was estimated to comprise,
                   in the case that htype is FLY_HOMOGRAPHY_RANSAC, a higher inlier_thr value will increase the
                   estimated inliers. Note that if the number of inliers is too low, it is likely
                   that a bad homography will be returned.
       \param[in]  x_src x coordinates of the source points.
       \param[in]  y_src y coordinates of the source points.
       \param[in]  x_dst x coordinates of the destination points.
       \param[in]  y_dst y coordinates of the destination points.
       \param[in]  htype can be FLY_HOMOGRAPHY_RANSAC, for which a RANdom SAmple Consensus will be
                   used to evaluate the homography quality (e.g., number of inliers), or FLY_HOMOGRAPHY_LMEDS,
                   which will use Least Median of Squares method to evaluate homography quality.
       \param[in]  inlier_thr if htype is FLY_HOMOGRAPHY_RANSAC, this parameter will five the maximum L2-distance
                   for a point to be considered an inlier.
       \param[in]  iterations maximum number of iterations when htype is FLY_HOMOGRAPHY_RANSAC and backend is CPU,
                   if backend is CUDA, iterations is the total number of iterations, an
                   iteration is a selection of 4 random points for which the homography is estimated
                   and evaluated for number of inliers.
       \param[in]  otype the array type for the homography output.
       \return     \ref FLY_SUCCESS if the computation is is successful,
                   otherwise an appropriate error code is returned.

       \ingroup cv_func_homography
     */
    FLY_API fly_err fly_homography(fly_array *H, int *inliers, const fly_array x_src, const fly_array y_src,
                               const fly_array x_dst, const fly_array y_dst,
                               const fly_homography_type htype, const float inlier_thr,
                               const unsigned iterations, const fly_dtype otype);

#ifdef __cplusplus
}
#endif
