cmake_minimum_required(VERSION 3.17)
project(RayTracer_GPU CUDA)

set(CMAKE_CUDA_STANDARD 14)


add_executable(RayTracer_GPU main.cu Vector3.cuh Ray.cuh util.cuh Hittable.cuh Camera.cuh Material.cuh)

set_target_properties(
        RayTracer_GPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)