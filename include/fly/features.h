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

typedef void * fly_features;

#ifdef __cplusplus
namespace fly
{
    class array;

    /// Represents a feature returned by a feature detector
    ///
    /// \ingroup flare_class
    /// \ingroup features_group_features
    class FLY_API features {
    private:
        fly_features feat;

    public:
        /// Default constructor. Creates a features object with new features
        features();

        /// Creates a features object with n features with undefined locations
        features(const size_t n);

        /// Creates a features object from a C fly_features object
        features(fly_features f);

        ~features();

        /// Copy assignment operator
        features& operator= (const features& other);

        /// Copy constructor
        features(const features &other);

#if FLY_COMPILER_CXX_RVALUE_REFERENCES
        /// Move constructor
        features(features &&other);

        /// Move assignment operator
        features &operator=(features &&other);
#endif

        /// Returns  the number of features represented by this object
        size_t getNumFeatures() const;

        /// Returns an fly::array which represents the x locations of a feature
        array getX() const;

        /// Returns an fly::array which represents the y locations of a feature
        array getY() const;

        /// Returns an array with the score of the features
        array getScore() const;

        /// Returns an array with the orientations of the features
        array getOrientation() const;

        /// Returns an array that represents the size of the features
        array getSize() const;

        /// Returns the underlying C fly_features object
        fly_features get() const;
    };

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /// Creates a new fly_feature object with \p num features
    ///
    /// \param[out] feat The new feature that will be created
    /// \param[in] num The number of features that will be in the new features
    ///                object
    /// \returns FLY_SUCCESS if successful
    /// \ingroup features_group_features
    FLY_API fly_err fly_create_features(fly_features *feat, dim_t num);

    /// Increases the reference count of the feature and all of its associated
    /// arrays
    ///
    /// \param[out] out The reference to the incremented array
    /// \param[in] feat The features object whose will be incremented
    ///                 object
    /// \returns FLY_SUCCESS if successful
    /// \ingroup features_group_features
    FLY_API fly_err fly_retain_features(fly_features *out, const fly_features feat);

    /// Returns the number of features associated with this object
    ///
    /// \param[out] num The number of features in the object
    /// \param[in] feat The feature whose count will be returned
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_num(dim_t *num, const fly_features feat);

    /// Returns the x positions of the features
    ///
    /// \param[out] out An array with all x positions of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_xpos(fly_array *out, const fly_features feat);

    /// Returns the y positions of the features
    ///
    /// \param[out] out An array with all y positions of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_ypos(fly_array *out, const fly_features feat);

    /// Returns the scores of the features
    ///
    /// \param[out] score An array with scores of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_score(fly_array *score, const fly_features feat);

    /// Returns the orientations of the features
    ///
    /// \param[out] orientation An array with the orientations of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_orientation(fly_array *orientation, const fly_features feat);

    /// Returns the size of the features
    ///
    /// \param[out] size An array with the sizes of the features
    /// \param[in] feat The features object
    /// \ingroup features_group_features
    FLY_API fly_err fly_get_features_size(fly_array *size, const fly_features feat);

    /// Reduces the reference count of each of the features
    ///
    /// \param[in] feat The features object whose reference count will be
    ///                 reduced
    /// \ingroup features_group_features
    FLY_API fly_err fly_release_features(fly_features feat);

#ifdef __cplusplus
}
#endif
