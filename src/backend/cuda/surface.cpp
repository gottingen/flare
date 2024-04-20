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
#include <GraphicsResourceManager.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <surface.hpp>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cuda {

template<typename T>
void copy_surface(const Array<T> &P, fg_surface surface) {
    auto stream = getActiveStream();
    if (DeviceManager::checkGraphicsInteropCapability()) {
        const T *d_P = P.get();

        auto res = interopManager().getSurfaceResources(surface);

        size_t bytes = 0;
        T *d_vbo     = NULL;
        cudaGraphicsMapResources(1, res[0].get(), stream);
        cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &bytes,
                                             *(res[0].get()));
        cudaMemcpyAsync(d_vbo, d_P, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaGraphicsUnmapResources(1, res[0].get(), stream);

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        TheiaModule &_ = theiaPlugin();
        unsigned bytes = 0, buffer = 0;
        THEIA_CHECK(_.fg_get_surface_vertex_buffer(&buffer, surface));
        THEIA_CHECK(_.fg_get_surface_vertex_buffer_size(&bytes, surface));

        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        auto *ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, P.get(), bytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T) \
    template void copy_surface<T>(const Array<T> &, fg_surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace cuda
}  // namespace flare
