//
// Created by igor on 16/08/2020.
//

#ifndef RAYTRACER_GPU_HITTABLE_CUH
#define RAYTRACER_GPU_HITTABLE_CUH

#include "util.cuh"

class Material;

struct hit_record {

    Point3 p;
    Vector3 normal;
    Material* mat_ptr;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& r, const Vector3& outward_normal){

        front_face = dot(r.direction(), outward_normal) < 0;

        normal = front_face ? outward_normal : -outward_normal;

    }

};

class Hittable {

    public:
        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const = 0;

};

class Sphere : public Hittable {

    public:
        __device__ Sphere() { }
        __device__ Sphere(Point3 c, float r, Material* m) : center(c) , radius(r), material(m) {}

        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;

    public:
        Point3 center;
        float radius;
        Material* material;

};

class HittableList : public Hittable {

    public:
        __device__ HittableList() { }
        __device__ HittableList(Hittable** l, int ls) {

            list = l;
            list_size = ls;
            list_current_size = 0;

            for(int i = 0; i < list_size; i++){

                if(list[i] != nullptr){

                    list_current_size++;

                }

            }

        }

        __device__ virtual bool hit(const Ray& r, float t_min, float t_max, hit_record& rec) const override;


    public:
        Hittable** list;
        int list_size;
        int list_current_size;

};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const {

    Vector3 oc = r.origin() - center;

    float a = r.direction().length_squared();
    float half_b = dot(oc,r.direction());
    float c = oc.length_squared() - radius*radius;

    float discriminant = half_b*half_b - a*c;

    if(discriminant > 0){

        float root = std::sqrt(discriminant);

        float temp = (-half_b - root)/a;

        if(temp < t_max && temp > t_min){

            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = material;
            return true;


        }

        temp = (-half_b + root)/a;

        if(temp < t_max && temp > t_min) {

            rec.t = temp;
            rec.p = r.at(rec.t);

            Vector3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = material;
            return true;
        }


    }

    return false;

}

__device__ bool HittableList::hit(const Ray& r, float t_min, float t_max, hit_record& rec) const {

    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for(int i = 0; i < list_current_size; i++){

        Hittable* object = *(list + i);

        if(object->hit(r, t_min, closest_so_far, temp_rec)){

            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;

        }

    }

    return hit_anything;

}

#endif //RAYTRACER_GPU_HITTABLE_CUH
