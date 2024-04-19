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
#include <surface.hpp>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cpu {

template<typename T>
void copy_surface(const Array<T> &P, fg_surface surface) {
    TheiaModule &_ = common::theiaPlugin();
    P.eval();
    getQueue().sync();

    CheckGL("Before CopyArrayToVBO");
    unsigned bytes = 0, buffer = 0;
    THEIA_CHECK(_.fg_get_surface_vertex_buffer(&buffer, surface));
    THEIA_CHECK(_.fg_get_surface_vertex_buffer_size(&bytes, surface));

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, P.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("In CopyArrayToVBO");
}

#define INSTANTIATE(T) \
    template void copy_surface<T>(const Array<T> &, fg_surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace flare
