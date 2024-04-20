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
#include <cstdio>
#include <iostream>

using namespace fly;

int main(int, char**) {
    try {
        static const float h_kernel[] = {1, 1, 1, 1, 0, 1, 1, 1, 1};
        static const int reset        = 500;
        static const int game_w = 128, game_h = 128;

        fly::info();

        std::cout << "This example demonstrates the Conway's Game of Life "
                     "using Flare"
                  << std::endl
                  << "There are 4 simple rules of Conways's Game of Life"
                  << std::endl
                  << "1. Any live cell with fewer than two live neighbours "
                     "dies, as if caused by under-population."
                  << std::endl
                  << "2. Any live cell with two or three live neighbours lives "
                     "on to the next generation."
                  << std::endl
                  << "3. Any live cell with more than three live neighbours "
                     "dies, as if by overcrowding."
                  << std::endl
                  << "4. Any dead cell with exactly three live neighbours "
                     "becomes a live cell, as if by reproduction."
                  << std::endl
                  << "Each white block in the visualization represents 1 alive "
                     "cell, black space represents dead cells"
                  << std::endl;

        fly::Window myWindow(512, 512, "Conway's Game of Life using Flare");

        int frame_count = 0;

        // Initialize the kernel array just once
        const fly::array kernel(3, 3, h_kernel, flyHost);
        array state;
        state = (fly::randu(game_h, game_w, f32) > 0.5).as(f32);

        while (!myWindow.close()) {
            myWindow.image(state);
            frame_count++;

            // Generate a random starting state
            if (frame_count % reset == 0)
                state = (fly::randu(game_h, game_w, f32) > 0.5).as(f32);

            // Convolve gets neighbors
            fly::array nHood = convolve(state, kernel);

            // Generate conditions for life
            // state == 1 && nHood < 2 ->> state = 0
            // state == 1 && nHood > 3 ->> state = 0
            // else if state == 1 ->> state = 1
            // state == 0 && nHood == 3 ->> state = 1
            fly::array C0 = (nHood == 2);
            fly::array C1 = (nHood == 3);

            // Update state
            state = state * C0 + C1;
        }
    } catch (fly::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }
    return 0;
}
