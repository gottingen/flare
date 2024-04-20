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

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <Array.hpp>
#include <common/graphics_common.hpp>
#include <err_cpu.hpp>
#include <image.hpp>
#include <platform.hpp>
#include <queue.hpp>

using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cpu {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    TheiaModule &_ = theiaPlugin();

    CheckGL("Before CopyArrayToImage");
    const T *d_X = in.get();
    getQueue().sync();

    unsigned data_size = 0, buffer = 0;
    THEIA_CHECK(_.fg_get_pixel_buffer(&buffer, image));
    THEIA_CHECK(_.fg_get_image_size(&data_size, image));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, data_size, d_X);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CheckGL("In CopyArrayToImage");
}

#define INSTANTIATE(T) template void copy_image<T>(const Array<T> &, fg_image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cpu
}  // namespace flare
