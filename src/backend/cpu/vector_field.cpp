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

#include <Array.hpp>
#include <common/graphics_common.hpp>
#include <err_cpu.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <vector_field.hpp>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cpu {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    TheiaModule &_ = theiaPlugin();
    points.eval();
    directions.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");

    unsigned size1 = 0, size2 = 0;
    unsigned buff1 = 0, buff2 = 0;
    THEIA_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
    THEIA_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
    THEIA_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
    THEIA_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

    glBindBuffer(GL_ARRAY_BUFFER, buff1);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size1, points.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, buff2);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size2, directions.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T)                                                     \
    template void copy_vector_field<T>(const Array<T> &, const Array<T> &, \
                                       fg_vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace flare
