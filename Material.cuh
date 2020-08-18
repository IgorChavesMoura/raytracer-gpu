//
// Created by igor on 17/08/2020.
//

#ifndef RAYTRACER_GPU_MATERIAL_CUH
#define RAYTRACER_GPU_MATERIAL_CUH

#include "util.cuh"

#include "Hittable.cuh"

__device__ Vector3 reflect(const Vector3& v, const Vector3& n){

    return v - 2*dot(v,n)*n;

}

__device__ Vector3 refract(const Vector3& uv, const Vector3& n, float etai_over_etat){

    float cos_theta = dot(-uv, n);

    Vector3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vector3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;

    return  r_out_perp + r_out_parallel;


}

__device__ float schlick(float cosine, float ref_idx){

    auto r0 = (1 - ref_idx) / (1 + ref_idx);

    r0 = r0*r0;

    return r0 + (1 - r0)*std::pow((1 - cosine), 5);

}


struct hit_record;

class Material {

    public:
        __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const = 0;

};

class Lambertian : public Material {

    public:
        __device__ Lambertian(const Color& a) : albedo(a) { }

        __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const {

            Vector3 scatter_direction = rec.normal + random_in_unit_sphere(rand_state);
            scattered = Ray(rec.p, scatter_direction);
            attenuation = albedo;

            return true;

        }

    public:
        Color albedo;
};

class Metal : public Material {

    public:
        __device__ Metal(const Color& a, const float f) : albedo(a), fuzz(f < 1 ? f : 1) { }

        __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const {

            Vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

            scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state));

            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);


        }

    public:
        Color albedo;
        float fuzz;

};

class Dielectric : public Material {

    public:
        __device__ Dielectric(float ri) : ref_idx(ri) { }

        __device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const {

            attenuation = Color(1.0, 1.0, 1.0);

            double etai_over_etat = rec.front_face ? (1.0/ref_idx) : ref_idx;

            Vector3 unit_direction = unit_vector(r_in.direction());

            double cos_theta = cu_min(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            if(etai_over_etat * sin_theta > 1.0){

                Vector3 reflected = reflect(unit_direction, rec.normal);
                scattered = Ray(rec.p, reflected);

                return true;

            }

            double reflect_prob = schlick(cos_theta, etai_over_etat);

            if(random_float(rand_state) < reflect_prob){

                Vector3 reflected = reflect(unit_direction, rec.normal);

                scattered = Ray(rec.p, reflected);

                return true;

            }

            Vector3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
            scattered = Ray(rec.p, refracted);

            return true;


        }

    public:
        float ref_idx;

};

#endif //RAYTRACER_GPU_MATERIAL_CUH
