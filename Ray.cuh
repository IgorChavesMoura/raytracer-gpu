//
// Created by igor on 15/08/2020.
//

#ifndef RAYTRACER_GPU_RAY_CUH
#define RAYTRACER_GPU_RAY_CUH


//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H

#include "Vector3.cuh"

class Ray {

public:

    __device__ Ray() { }
    __device__ Ray(const Point3& origin, const Vector3& direction) : orig(origin), dir(direction) { }

    __device__ Point3 origin() const {

        return orig;

    }

    __device__ Vector3 direction() const {

        return dir;

    }

    __device__ Point3 at(float t) const {

        return orig + t*dir;

    }


public:
    Point3 orig;
    Vector3 dir;

};


#endif //RAYTRACER_RAY_H



#endif //RAYTRACER_GPU_RAY_CUH
