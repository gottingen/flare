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
#include <GraphicsResourceManager.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <image.hpp>

using fly::dim4;
using flare::common::TheiaManager;
using flare::common::TheiaModule;
using flare::common::theiaPlugin;

namespace flare {
namespace cuda {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    auto stream = getActiveStream();
    if (DeviceManager::checkGraphicsInteropCapability()) {
        auto res = interopManager().getImageResources(image);

        const T *d_X = in.get();
        size_t bytes = 0;
        T *d_pixels  = NULL;
        cudaGraphicsMapResources(1, res[0].get(), stream);
        cudaGraphicsResourceGetMappedPointer((void **)&d_pixels, &bytes,
                                             *(res[0].get()));
        cudaMemcpyAsync(d_pixels, d_X, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaGraphicsUnmapResources(1, res[0].get(), stream);

        POST_LAUNCH_CHECK();
        CheckGL("After cuda resource copy");
    } else {
        TheiaModule &_ = common::theiaPlugin();
        CheckGL("Begin CUDA fallback-resource copy");
        unsigned data_size = 0, buffer = 0;
        THEIA_CHECK(_.fg_get_image_size(&data_size, image));
        THEIA_CHECK(_.fg_get_pixel_buffer(&buffer, image));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, 0, GL_STREAM_DRAW);
        auto *ptr = static_cast<GLubyte *>(
            glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, in.get(), data_size,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
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

}  // namespace cuda
}  // namespace flare
