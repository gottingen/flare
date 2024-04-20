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

#include <algorithm>
#include <sstream>
#include <utility>
#include "../common/idxio.h"

bool compare(const std::pair<float, int> l, const std::pair<float, int> r) {
    return l.first >= r.first;
}

typedef std::pair<float, int> sort_type;

template<bool expand_labels>
std::string classify(fly::array arr, int k) {
    std::stringstream ss;
    if (expand_labels) {
        fly::array vec = arr(fly::span, k).as(f32);
        float *h_vec  = vec.host<float>();
        std::vector<sort_type> data;

        for (int i = 0; i < (int)vec.elements(); i++)
            data.push_back(std::make_pair(h_vec[i], i));

        std::stable_sort(data.begin(), data.end(), compare);

        fly::freeHost(h_vec);
        ss << data[0].second;
    } else {
        ss << (int)(arr(k).as(f32).scalar<float>());
    }
    return ss.str();
}

template<bool expand_labels>
static void setup_mnist(int *num_classes, int *num_train, int *num_test,
                        fly::array &train_images, fly::array &test_images,
                        fly::array &train_labels, fly::array &test_labels,
                        float frac) {
    std::vector<dim_t> idims;
    std::vector<float> idata;
    read_idx(idims, idata, ASSETS_DIR "/examples/data/mnist/images-subset");

    std::vector<dim_t> ldims;
    std::vector<unsigned> ldata;
    read_idx(ldims, ldata, ASSETS_DIR "/examples/data/mnist/labels-subset");

    std::reverse(idims.begin(), idims.end());
    unsigned numdims = idims.size();
    fly::array images = fly::array(fly::dim4(numdims, &idims[0]), &idata[0]);

    fly::array R             = fly::randu(10000, 1);
    fly::array cond          = R < std::min(frac, 0.8f);
    fly::array train_indices = where(cond);
    fly::array test_indices  = where(!cond);

    train_images = lookup(images, train_indices, 2) / 255;
    test_images  = lookup(images, test_indices, 2) / 255;

    *num_classes = 10;
    *num_train   = train_images.dims(2);
    *num_test    = test_images.dims(2);

    if (expand_labels) {
        train_labels = fly::constant(0, *num_classes, *num_train);
        test_labels  = fly::constant(0, *num_classes, *num_test);

        unsigned *h_train_idx = train_indices.host<unsigned>();
        unsigned *h_test_idx  = test_indices.host<unsigned>();

        for (int ii = 0; ii < *num_train; ii++) {
            train_labels(ldata[h_train_idx[ii]], ii) = 1;
        }

        for (int ii = 0; ii < *num_test; ii++) {
            test_labels(ldata[h_test_idx[ii]], ii) = 1;
        }

        fly::freeHost(h_train_idx);
        fly::freeHost(h_test_idx);
    } else {
        fly::array labels = fly::array(ldims[0], &ldata[0]);
        train_labels     = labels(train_indices);
        test_labels      = labels(test_indices);
    }

    return;
}

#if 0
static fly::array randidx(int num, int total)
{
    fly::array locs;
    do {
        locs = fly::where(fly::randu(total, 1) < float(num * 2) / total);
    } while (locs.elements() < num);

    return locs(fly::seq(num));
}
#endif

template<bool expand_labels>
static void display_results(const fly::array &test_images,
                            const fly::array &test_output,
                            const fly::array &test_actual, int num_display) {
#if 0
    fly::array locs = randidx(num_display, test_images.dims(2));

    fly::array disp_in  = test_images(fly::span, fly::span, locs);
    fly::array disp_out = expand_labels ? test_output(fly::span, locs) : test_output(locs);

    for (int i = 0; i < 5; i++) {

        int imgs_per_iter = num_display / 5;
        for (int j = 0; j < imgs_per_iter; j++) {

            int k = i * imgs_per_iter + j;
            fly::fig("sub", 2, imgs_per_iter / 2, j+1);

            fly::image(disp_in(fly::span, fly::span, k).T());
            std::string pred_name = std::string("Predicted: ");
            pred_name = pred_name + classify<expand_labels>(disp_out, k);
            fly::fig("title", pred_name.c_str());
        }

        printf("Press any key to see next set");
        getchar();
    }
#else
    using namespace fly;
    for (int i = 0; i < num_display; i++) {
        std::cout << "Predicted: " << classify<expand_labels>(test_output, i)
                  << std::endl;
        std::cout << "Actual: " << classify<expand_labels>(test_actual, i)
                  << std::endl;

        unsigned char *img =
            (test_images(span, span, i) > 0.1f).as(u8).host<unsigned char>();
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                std::cout << (img[j * 28 + k] ? "\u2588" : " ") << " ";
            }
            std::cout << std::endl;
        }
        fly::freeHost(img);
        getchar();
    }
#endif
}
