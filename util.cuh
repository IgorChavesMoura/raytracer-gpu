//
// Created by igor on 15/08/2020.
//

#ifndef RAYTRACER_GPU_UTIL_CUH
#define RAYTRACER_GPU_UTIL_CUH

#include <iostream>
#include <cmath>
#include <float.h>

#include <curand_kernel.h>

#include "Vector3.cuh"
#include "Ray.cuh"

const float infinity = FLT_MAX;
const float pi = 3.1415926535897932385f;

__device__ float cu_min(float a, float b) {

    return a <= b ? a : b;

}

__device__ float degrees_to_radians(float degrees) {

    return degrees * pi/180.0f;

}

__device__ float random_float(curandState* rand_state){

    return curand_uniform(rand_state);

}

__device__ float random_float(curandState* rand_state, float min, float max){

    return min + (max-min)*random_float(rand_state);

}



#define RANDVEC3 Vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vector3 random_in_unit_sphere(curandState* local_rand_state){

    Vector3 v;

    do {

        v = 2.0f * RANDVEC3 - Vector3(1.0f, 1.0f, 1.0f);

    } while (v.length_squared() >= 1.0f);

    return v;

}

__device__ Vector3 random_in_unit_disk(curandState* local_rand_state) {
    Vector3 v;

    do {

        v = 2.0f * Vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - Vector3(1.0f, 1.0f, 0.0f);

    } while (dot(v, v) >= 1.0f);

    return v;
}


void write_buffer(Color* frame_buffer, int nx, int ny){

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = ny - 1; i >= 0; i--) {
        for (int j = 0; j < nx; j++) {

            size_t pixel_index = i * nx + j;

            float r = frame_buffer[pixel_index].r();
            float g = frame_buffer[pixel_index].g();
            float b = frame_buffer[pixel_index].b();

            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);

            std::cout << ir << " " << ig << " " << ib << "\n";

        }
    }

}



#endif //RAYTRACER_GPU_UTIL_CUH
