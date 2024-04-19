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
#include <image.hpp>

#include <stdexcept>
#include <vector>

using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace opencl {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    TheiaModule &_ = theiaPlugin();
    if (isGLSharingSupported()) {
        CheckGL("Begin opencl resource copy");

        auto res = interopManager().getImageResources(image);

        const cl::Buffer *d_X = in.get();

        unsigned bytes = 0;
        THEIA_CHECK(_.fg_get_image_size(&bytes, image));

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*(res[0].get()));

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_X, *(res[0].get()), 0, 0, bytes, NULL,
                                     &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End opencl resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        unsigned bytes = 0, buffer = 0;
        THEIA_CHECK(_.fg_get_image_size(&bytes, image));
        THEIA_CHECK(_.fg_get_pixel_buffer(&buffer, image));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, 0, GL_STREAM_DRAW);
        auto *ptr = static_cast<GLubyte *>(
            glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            getQueue().enqueueReadBuffer(*in.get(), CL_TRUE, 0, bytes, ptr);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
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

}  // namespace opencl
}  // namespace flare
