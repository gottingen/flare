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

#include <flare.h>
#include <math.h>
#include <stdio.h>
#include <fly/util.h>
#include <string>
#include <vector>
#include "mnist_common.h"

using namespace fly;

// Get accuracy of the predicted results
float accuracy(const array &predicted, const array &target) {
    return 100 * count<float>(predicted == target) / target.elements();
}

// Calculate all the distances from testing set to training set
array distance(array train, array test) {
    const int feat_len  = train.dims(1);
    const int num_train = train.dims(0);
    const int num_test  = test.dims(0);
    array dist          = constant(0, num_train, num_test);

    // Iterate over each attribute
    for (int ii = 0; ii < feat_len; ii++) {
        // Get a attribute vectors
        array train_i = train(span, ii);
        array test_i  = test(span, ii).T();

        // Tile the vectors to generate matrices
        array train_tiled = tile(train_i, 1, num_test);
        array test_tiled  = tile(test_i, num_train, 1);

        // Add the distance for this attribute
        dist = dist + abs(train_tiled - test_tiled);
        dist.eval();  // Necessary to free up train_i, test_i
    }

    return dist;
}

array knn(array &train_feats, array &test_feats, array &train_labels) {
    // Find distances between training and testing sets
    array dist = distance(train_feats, test_feats);

    // Find the neighbor producing the minimum distance
    array val, idx;
    min(val, idx, dist);

    // Return the labels
    return train_labels(idx);
}

array bagging(array &train_feats, array &test_feats, array &train_labels,
              int num_classes, int num_models, int sample_size) {
    int num_train = train_feats.dims(0);
    int num_test  = test_feats.dims(0);

    array idx        = floor(randu(sample_size, num_models) * num_train);
    array labels_all = constant(0, num_test, num_classes);
    array off        = seq(num_test);

    for (int i = 0; i < num_models; i++) {
        array ii = idx(span, i);

        array train_feats_ii  = lookup(train_feats, ii, 0);
        array train_labels_ii = train_labels(ii);

        // Get the predicted results
        array labels_ii = knn(train_feats_ii, test_feats, train_labels_ii);
        array lidx      = labels_ii * num_test + off;

        labels_all(lidx) = labels_all(lidx) + 1;
    }

    array val, labels;
    max(val, labels, labels_all, 1);

    return labels;
}

void bagging_demo(bool console, int perc) {
    array train_images, train_labels;
    array test_images, test_labels;
    int num_train, num_test, num_classes;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<false>(&num_classes, &num_train, &num_test, train_images,
                       test_images, train_labels, test_labels, frac);

    int feature_length = train_images.elements() / num_train;
    array train_feats  = moddims(train_images, feature_length, num_train).T();
    array test_feats   = moddims(test_images, feature_length, num_test).T();

    int num_models  = 10;
    int sample_size = 1000;

    timer::start();
    // Get the predicted results
    array res_labels = bagging(train_feats, test_feats, train_labels,
                               num_classes, num_models, sample_size);
    double test_time = timer::stop();

    // Results
    printf("Accuracy on testing  data: %2.2f\n",
           accuracy(res_labels, test_labels));

    printf("Prediction time: %4.4f\n", test_time);

    if (false && !console) {
        display_results<false>(test_images, res_labels, test_labels.T(), 20);
    }
}

int main(int argc, char **argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;

    try {
        setDevice(device);
        fly::info();
        bagging_demo(console, perc);

    } catch (fly::exception &ae) { std::cerr << ae.what() << std::endl; }

    return 0;
}
