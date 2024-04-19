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
#include <plot.hpp>

using fly::dim4;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace opencl {

template<typename T>
void copy_plot(const Array<T> &P, fg_plot plot) {
    TheiaModule &_ = theiaPlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_P = P.get();
        unsigned bytes        = 0;
        THEIA_CHECK(_.fg_get_plot_vertex_buffer_size(&bytes, plot));

        auto res = interopManager().getPlotResources(plot);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_P, *(res[0].get()), 0, 0, bytes, NULL,
                                     &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        unsigned bytes = 0, buffer = 0;
        THEIA_CHECK(_.fg_get_plot_vertex_buffer(&buffer, plot));
        THEIA_CHECK(_.fg_get_plot_vertex_buffer_size(&bytes, plot));

        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        auto *ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            getQueue().enqueueReadBuffer(*P.get(), CL_TRUE, 0, bytes, ptr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T) template void copy_plot<T>(const Array<T> &, fg_plot);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace opencl
}  // namespace flare
