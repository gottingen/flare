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
#include <vector_field.hpp>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cuda {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    auto stream = getActiveStream();
    if (DeviceManager::checkGraphicsInteropCapability()) {
        auto res = interopManager().getVectorFieldResources(vfield);
        cudaGraphicsResource_t resources[2] = {*res[0].get(), *res[1].get()};

        cudaGraphicsMapResources(2, resources, stream);

        // Points
        {
            const T *ptr = points.get();
            size_t bytes = 0;
            T *d_vbo     = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &bytes,
                                                 resources[0]);
            cudaMemcpyAsync(d_vbo, ptr, bytes, cudaMemcpyDeviceToDevice,
                            stream);
        }
        // Directions
        {
            const T *ptr = directions.get();
            size_t bytes = 0;
            T *d_vbo     = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &bytes,
                                                 resources[1]);
            cudaMemcpyAsync(d_vbo, ptr, bytes, cudaMemcpyDeviceToDevice,
                            stream);
        }
        cudaGraphicsUnmapResources(2, resources, stream);

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        TheiaModule &_ = theiaPlugin();
        CheckGL("Begin CUDA fallback-resource copy");
        unsigned size1 = 0, size2 = 0;
        unsigned buff1 = 0, buff2 = 0;
        THEIA_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
        THEIA_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
        THEIA_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
        THEIA_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

        // Points
        glBindBuffer(GL_ARRAY_BUFFER, buff1);
        auto *ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, points.get(), size1,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Directions
        glBindBuffer(GL_ARRAY_BUFFER, buff2);
        ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, directions.get(), size2,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)                                                     \
    template void copy_vector_field<T>(const Array<T> &, const Array<T> &, \
                                       fg_vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace cuda
}  // namespace flare
