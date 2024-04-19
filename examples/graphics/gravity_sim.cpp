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
#include <vector>
#include "gravity_sim_init.h"

using namespace fly;
using namespace std;

static const bool is3D           = true;
const static int total_particles = 4000;
static const int reset           = 3000;
static const float min_dist      = 3;
static const int width = 768, height = 768, depth = 768;
static const int gravity_constant = 20000;

float mass_range = 0;
float min_mass   = 0;

void initial_conditions_rand(fly::array &mass, vector<fly::array> &pos,
                             vector<fly::array> &vels,
                             vector<fly::array> &forces) {
    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i]    = fly::randn(total_particles) * width + width;
        vels[i]   = 0 * fly::randu(total_particles) - 0.5;
        forces[i] = fly::constant(0, total_particles);
    }
    mass = fly::constant(gravity_constant, total_particles);
}

void initial_conditions_galaxy(fly::array &mass, vector<fly::array> &pos,
                               vector<fly::array> &vels,
                               vector<fly::array> &forces) {
    fly::array initial_cond_consts(fly::dim4(7, total_particles), hbd);
    initial_cond_consts = initial_cond_consts.T();

    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i]    = fly::randn(total_particles) * width + width;
        vels[i]   = 0 * (fly::randu(total_particles) - 0.5);
        forces[i] = fly::constant(0, total_particles);
    }

    mass    = initial_cond_consts(span, 0);
    pos[0]  = (initial_cond_consts(span, 1) / 32 + 0.6) * width;
    pos[1]  = (initial_cond_consts(span, 2) / 32 + 0.3) * height;
    pos[2]  = (initial_cond_consts(span, 3) / 32 + 0.5) * depth;
    vels[0] = (initial_cond_consts(span, 4) / 32) * width;
    vels[1] = (initial_cond_consts(span, 5) / 32) * height;
    vels[2] = (initial_cond_consts(span, 6) / 32) * depth;

    pos[0](seq(0, pos[0].dims(0) - 1, 2)) -= 0.4 * width;
    pos[1](seq(0, pos[0].dims(0) - 1, 2)) += 0.4 * height;
    vels[0](seq(0, pos[0].dims(0) - 1, 2)) += 4;

    min_mass   = min<float>(mass);
    mass_range = max<float>(mass) - min<float>(mass);
}

fly::array ids_from_pos(vector<fly::array> &pos) {
    return (pos[0].as(u32) * height) + pos[1].as(u32);
}

fly::array ids_from_3D(vector<fly::array> &pos, float Rx, float Ry, float Rz) {
    fly::array x0 = (pos[0] - width / 2);
    fly::array y0 =
        (pos[1] - height / 2) * cos(Rx) + (pos[2] - depth / 2) * sin(Rx);
    fly::array z0 =
        (pos[2] - depth / 2) * cos(Rx) - (pos[2] - depth / 2) * sin(Rx);

    fly::array x1 = x0 * cos(Ry) - z0 * sin(Ry);
    fly::array y1 = y0;

    fly::array x2 = x1 * cos(Rz) + y1 * sin(Rz);
    fly::array y2 = y1 * cos(Rz) - x1 * sin(Rz);

    x2 += width / 2;
    y2 += height / 2;

    return (x2.as(u32) * height) + y2.as(u32);
}

fly::array ids_from_3D(vector<fly::array> &pos, float Rx, float Ry, float Rz,
                      fly::array filter) {
    fly::array x0 = (pos[0](filter) - width / 2);
    fly::array y0 = (pos[1](filter) - height / 2) * cos(Rx) +
                   (pos[2](filter) - depth / 2) * sin(Rx);
    fly::array z0 = (pos[2](filter) - depth / 2) * cos(Rx) -
                   (pos[2](filter) - depth / 2) * sin(Rx);

    fly::array x1 = x0 * cos(Ry) - z0 * sin(Ry);
    fly::array y1 = y0;

    fly::array x2 = x1 * cos(Rz) + y1 * sin(Rz);
    fly::array y2 = y1 * cos(Rz) - x1 * sin(Rz);

    x2 += width / 2;
    y2 += height / 2;

    return (x2.as(u32) * height) + y2.as(u32);
}

void simulate(fly::array &mass, vector<fly::array> &pos, vector<fly::array> &vels,
              vector<fly::array> &forces, float dt) {
    for (int i = 0; i < (int)pos.size(); ++i) {
        pos[i] += vels[i] * dt;
        pos[i].eval();
    }

    // calculate forces to each particle
    vector<fly::array> diff(pos.size());
    fly::array dist = fly::constant(0, pos[0].dims(0), pos[0].dims(0));

    for (int i = 0; i < (int)pos.size(); ++i) {
        diff[i] = tile(pos[i], 1, pos[i].dims(0)) -
                  transpose(tile(pos[i], 1, pos[i].dims(0)));
        dist += (diff[i] * diff[i]);
    }

    dist = sqrt(dist);
    dist = fly::max(min_dist, dist);
    dist *= dist * dist;

    for (int i = 0; i < (int)pos.size(); ++i) {
        // calculate force vectors
        forces[i] = diff[i] / dist;
        forces[i].eval();

        // fly::array idx = fly::where(fly::isNaN(forces[i]));
        // if(idx.elements() > 0)
        //    forces[i](idx) = 0;
        // forces[i] = sum(forces[i]).T();
        forces[i] = matmul(forces[i].T(), mass);

        // update force scaled to time, magnitude constant
        forces[i] *= (gravity_constant);
        forces[i].eval();

        // update velocities from forces
        vels[i] += forces[i] * dt;
        vels[i].eval();

        // noise
        // forces[i] += 0.1 * fly::randn(forces[i].dims(0));

        // dampening
        // vels[i] *= 1 - (0.005*dt);
    }
}

void collisions(vector<fly::array> &pos, vector<fly::array> &vels, bool is3D) {
    // clamp particles inside screen border
    fly::array invalid_x = -2 * (pos[0] > width - 1 || pos[0] < 0) + 1;
    fly::array invalid_y = -2 * (pos[1] > height - 1 || pos[1] < 0) + 1;
    // fly::array invalid_x = (pos[0] < width-1 || pos[0] > 0);
    // fly::array invalid_y = (pos[1] < height-1 || pos[1] > 0);
    vels[0] = invalid_x * vels[0];
    vels[1] = invalid_y * vels[1];

    fly::array projected_px = min(width - 1, max(0, pos[0]));
    fly::array projected_py = min(height - 1, max(0, pos[1]));
    pos[0]                 = projected_px;
    pos[1]                 = projected_py;

    if (is3D) {
        fly::array invalid_z    = -2 * (pos[2] > depth - 1 || pos[2] < 0) + 1;
        vels[2]                = invalid_z * vels[2];
        fly::array projected_pz = min(depth - 1, max(0, pos[2]));
        pos[2]                 = projected_pz;
    }
}

int main(int, char **) {
    try {
        fly::info();

        fly::Window myWindow(width, height,
                            "Gravity Simulation using Flare");
        myWindow.setColorMap(FLY_COLORMAP_HEAT);

        int frame_count = 0;

        // Initialize the kernel array just once
        const fly::array draw_kernel = gaussianKernel(7, 7);

        const int dims = (is3D) ? 3 : 2;

        vector<fly::array> pos(dims);
        vector<fly::array> vels(dims);
        vector<fly::array> forces(dims);
        fly::array mass;

        // Generate a random starting state
        initial_conditions_galaxy(mass, pos, vels, forces);

        fly::array image = fly::constant(0, width, height);
        fly::array ids(total_particles, u32);

        fly::timer timer = fly::timer::start();
        while (!myWindow.close()) {
            float dt = fly::timer::stop(timer);
            timer    = fly::timer::start();

            fly::array mid = mass(span) > (min_mass + mass_range / 3);
            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0, mid)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0, mid) : ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            mid = mass(span) > (min_mass + 2 * mass_range / 3);
            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0, mid)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0, mid) : ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            ids = (is3D) ? ids_from_3D(pos, 0, 0 + frame_count / 150.f, 0)
                         : ids_from_pos(pos);
            // ids = (is3D)? ids_from_3D(pos, 0, 0, 0) :  ids_from_pos(pos);
            // //uncomment for no 3d rotation
            image(ids) += 4.f;

            image = convolve(image, draw_kernel);
            myWindow.image(image);
            image = fly::constant(0, image.dims());

            frame_count++;

            // Generate a random starting state
            if (frame_count % reset == 0) {
                initial_conditions_galaxy(mass, pos, vels, forces);
            }

            // simulate
            simulate(mass, pos, vels, forces, dt);

            // check for collisions and adjust positions/velocities accordingly
            collisions(pos, vels, is3D);
        }
    } catch (fly::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
