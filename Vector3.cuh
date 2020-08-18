//
// Created by igor on 15/08/2020.
//

#ifndef RAYTRACER_GPU_VECTOR3_CUH
#define RAYTRACER_GPU_VECTOR3_CUH


//
// Created by igor on 05/08/2020.
//

#ifndef RAYTRACER_VECTOR3_H
#define RAYTRACER_VECTOR3_H

#include <iostream>

class Vector3 {

public:

    __host__ __device__ Vector3() : e{0.0f, 0.0f, 0.0f} {}
    __host__ __device__ Vector3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const {
        return e[0];
    }

    __host__ __device__ float y() const {
        return e[1];
    }

    __host__ __device__ float z() const {
        return e[2];
    }

    __host__ __device__ float r() const {
        return e[0];
    }

    __host__ __device__ float g() const {
        return e[1];
    }

    __host__ __device__ float b() const {
        return e[2];
    }

    __host__ __device__ float length() const {

        return std::sqrt(length_squared());

    }


    __host__ __device__ float length_squared() const {

        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];

    }

    __host__ __device__ Vector3 operator-() const {
        return Vector3(-e[0], -e[1], -e[2]);
    }

    __host__ __device__ float operator[](int i) const {

        return e[i];

    }

    __host__ __device__ float& operator[](int i){

        return e[i];

    }

    __host__ __device__ Vector3& operator+=(const Vector3& v){

        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];

        return *this;

    }

    __host__ __device__ Vector3& operator-=(const Vector3& v){

        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];

        return *this;
    }

    __host__ __device__ Vector3& operator*=(const float t){

        e[0] *= t;
        e[1] *= t;
        e[2] *= t;

        return *this;

    }

    __host__ __device__ Vector3& operator*=(const Vector3& v){

        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];

        return *this;

    }

    __host__ __device__ Vector3& operator/=(const float t){

        return *this *= 1/t;

    }




public:
    float e[3];

};


inline std::ostream& operator<<(std::ostream& out, const Vector3& v){

    return out << '[' << v.e[0] << ", " << v.e[1] << ", " << v.e[2] << ']';

}

__host__ __device__ inline Vector3 operator+(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);

}

__host__ __device__ inline Vector3 operator-(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);

}

__host__ __device__ inline Vector3 operator*(const Vector3& u, const Vector3& v){

    return Vector3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);

}

__host__ __device__ inline Vector3 operator*(const float t, const Vector3& v){

    return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);

}

__host__ __device__ inline Vector3 operator*(const Vector3& v, const float t){

    return t * v;

}

__host__ __device__ inline Vector3 operator/(const Vector3& v, const float t){

    return (1/t) * v;

}


__host__ __device__ inline float dot(const Vector3& u, const Vector3& v){

    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];

}

__host__ __device__ inline Vector3 cross(const Vector3& u, const Vector3& v){

    return Vector3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                   u.e[2] * v.e[0] - u.e[0] * v.e[2],
                   u.e[0] * v.e[1] - u.e[1] * v.e[0]);

}

__host__ __device__ inline Vector3 unit_vector(Vector3 v){

    return v / v.length();

}




using Point3 = Vector3;
using Color = Vector3;

#endif //RAYTRACER_VECTOR3_H



#endif //RAYTRACER_GPU_VECTOR3_CUH
