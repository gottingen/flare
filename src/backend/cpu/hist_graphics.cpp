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

#include <err_cpu.hpp>
#include <hist_graphics.hpp>
#include <platform.hpp>
#include <queue.hpp>

using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cpu {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist) {
    TheiaModule &_ = theiaPlugin();
    data.eval();
    getQueue().sync();

    CheckGL("Begin copy_histogram");
    unsigned bytes = 0, buffer = 0;
    THEIA_CHECK(_.fg_get_histogram_vertex_buffer(&buffer, hist));
    THEIA_CHECK(_.fg_get_histogram_vertex_buffer_size(&bytes, hist));

    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data.get());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    CheckGL("End copy_histogram");
}

#define INSTANTIATE(T) \
    template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cpu
}  // namespace flare
