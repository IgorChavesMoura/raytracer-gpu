//
// Created by igor on 16/08/2020.
//

#ifndef RAYTRACER_GPU_CAMERA_CUH
#define RAYTRACER_GPU_CAMERA_CUH

#include "util.cuh"

class Camera {

    public:
        __device__ Camera(Point3 lookfrom, Point3 lookat, Vector3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist){


            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2);
            float viewport_height = 2.0f * h;
            float viewport_width  = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            len_radius = aperture / 2;


        }

        __device__ Ray get_ray(float s, float t, curandState* rand_state) {

            Vector3 rd = len_radius * random_in_unit_disk(rand_state);
            Vector3 offset = u * rd.x() + v * rd.y();


            return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);

        }

    public:
        Point3 origin;
        Point3 lower_left_corner;

        Vector3 horizontal;
        Vector3 vertical;

        Vector3 u, v, w;

        float len_radius;

};


#endif //RAYTRACER_GPU_CAMERA_CUH
