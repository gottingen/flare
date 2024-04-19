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
#include <debug_opencl.hpp>
#include <err_opencl.hpp>
#include <vector_field.hpp>

using fly::dim4;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace opencl {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    TheiaModule &_ = common::theiaPlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_points     = points.get();
        const cl::Buffer *d_directions = directions.get();
        unsigned pBytes                = 0;
        unsigned dBytes                = 0;
        THEIA_CHECK(_.fg_get_vector_field_vertex_buffer_size(&pBytes, vfield));
        THEIA_CHECK(_.fg_get_vector_field_direction_buffer_size(&dBytes, vfield));

        auto res = interopManager().getVectorFieldResources(vfield);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));
        shared_objects.push_back(*(res[1].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_points, *(res[0].get()), 0, 0, pBytes,
                                     NULL, &event);
        getQueue().enqueueCopyBuffer(*d_directions, *(res[1].get()), 0, 0,
                                     dBytes, NULL, &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        unsigned size1 = 0, size2 = 0;
        unsigned buff1 = 0, buff2 = 0;
        THEIA_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
        THEIA_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
        THEIA_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
        THEIA_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

        CheckGL("Begin OpenCL fallback-resource copy");

        // Points
        glBindBuffer(GL_ARRAY_BUFFER, buff1);
        auto *pPtr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (pPtr) {
            getQueue().enqueueReadBuffer(*points.get(), CL_TRUE, 0, size1,
                                         pPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Directions
        glBindBuffer(GL_ARRAY_BUFFER, buff2);
        auto *dPtr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (dPtr) {
            getQueue().enqueueReadBuffer(*directions.get(), CL_TRUE, 0, size2,
                                         dPtr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
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

}  // namespace opencl
}  // namespace flare
