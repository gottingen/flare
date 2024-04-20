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
#include <common/defines.hpp>
#include <fly/device.h>
#include <fly/dim4.hpp>
#include <cstddef>
#include <vector>

fly::dim4 calcStrides(const fly::dim4& parentDim);

fly::dim4 getOutDims(const fly::dim4& ldims, const fly::dim4& rdims,
                    bool batchMode);

/// Array Arrayementation Info class
// This class is the base class to all Array objects. The purpose of this class
// was to have a way to retrieve basic information of an Array object without
// specifying what type the object is at compile time.
class ArrayInfo {
   private:
    // The devId variable stores information about the deviceId as well as the
    // backend. The 8 LSBs (0-7) are used to store the device ID. The 09th LSB
    // is set to 1 if backend is CPU The 10th LSB is set to 1 if backend is CUDA
    //
    // This information can be retrieved directly from an fly_array by doing
    //     int* devId = reinterpret_cast<int*>(a); // a is an fly_array
    //     fly_backend backendID = *devId >> 8;   // Returns 1, 2, 4 for CPU,
    //     CUDA  respectively int        deviceID  = *devId & 0xff; //
    //     Returns devices ID between 0-255
    // This is possible by doing a static_assert on devId
    //
    // This can be changed in the future if the need arises for more devices as
    // this implementation is internal. Make sure to change the bit shift ops
    // when such a change is being made
    unsigned devId;
    fly_dtype type;
    fly::dim4 dim_size;
    dim_t offset;
    fly::dim4 dim_strides;
    bool is_sparse;

   public:
    ArrayInfo(unsigned id, fly::dim4 size, dim_t offset_, fly::dim4 stride,
              fly_dtype fly_type);

    ArrayInfo(unsigned id, fly::dim4 size, dim_t offset_, fly::dim4 stride,
              fly_dtype fly_type, bool sparse);

    ArrayInfo()                       = default;
    ArrayInfo(const ArrayInfo& other) = default;
    ArrayInfo(ArrayInfo&& other)      = default;

    ArrayInfo& operator=(ArrayInfo other) noexcept {
        swap(other);
        return *this;
    }

    void swap(ArrayInfo& other) noexcept {
        using std::swap;
        swap(devId, other.devId);
        swap(type, other.type);
        swap(dim_size, other.dim_size);
        swap(offset, other.offset);
        swap(dim_strides, other.dim_strides);
        swap(is_sparse, other.is_sparse);
    }

    const fly_dtype& getType() const { return type; }

    dim_t getOffset() const { return offset; }

    const fly::dim4& strides() const { return dim_strides; }

    dim_t elements() const { return dim_size.elements(); }
    dim_t ndims() const { return dim_size.ndims(); }
    const fly::dim4& dims() const { return dim_size; }
    size_t total() const { return offset + dim_strides[3] * dim_size[3]; }

    unsigned getDevId() const;

    void setId(int id) const;

    void setId(int id);

    fly_backend getBackendId() const;

    void resetInfo(const fly::dim4& dims) {
        dim_size    = dims;
        dim_strides = calcStrides(dims);
        offset      = 0;
    }

    void resetDims(const fly::dim4& dims) { dim_size = dims; }

    void modDims(const fly::dim4& newDims);

    void modStrides(const fly::dim4& newStrides);

    bool isEmpty() const;

    bool isScalar() const;

    bool isRow() const;

    bool isColumn() const;

    bool isVector() const;

    bool isComplex() const;

    bool isReal() const;

    bool isDouble() const;

    bool isSingle() const;

    bool isHalf() const;

    bool isRealFloating() const;

    bool isFloating() const;

    bool isInteger() const;

    bool isBool() const;

    bool isLinear() const;

    bool isSparse() const;
};

fly::dim4 toDims(const std::vector<fly_seq>& seqs, const fly::dim4& parentDims);

fly::dim4 toOffset(const std::vector<fly_seq>& seqs, const fly::dim4& parentDims);

fly::dim4 toStride(const std::vector<fly_seq>& seqs, const fly::dim4& parentDims);
