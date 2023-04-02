#pragma once

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "texture.h"


class constant_medium : public hittable {
public:
    __device__
    constant_medium(hittable* b, double d, texture* a)
        : boundary(b),
        neg_inv_density(-1 / d),
        phase_function(new isotropic(a))
    {}

    __device__
    constant_medium(hittable* b, double d, color c)
        : boundary(b),
        neg_inv_density(-1 / d),
        phase_function(new isotropic(c))
    {}

    __device__
    virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const override;

    __device__
    virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
        return boundary->bounding_box(time0, time1, output_box);
    }

    __device__
    ~constant_medium() { delete phase_function; }

public:
    hittable* boundary;
    material* phase_function;
    double neg_inv_density;
};

__device__
bool constant_medium::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    //const bool enableDebug = false;
    //const bool debugging = enableDebug && random_double() < 0.00001;

    hit_record rec1, rec2;

    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2))
        return false;

    // if (debugging) std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = r.direction().length();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    //Need randomness here to actually simulate the medium, but this can wait
    const auto hit_distance = neg_inv_density * -0.3;

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    /*
    if (debugging) {
        std::cerr << "hit_distance = " << hit_distance << '\n'
            << "rec.t = " << rec.t << '\n'
            << "rec.p = " << rec.p << '\n';
    }*/

    rec.normal = vec3(1, 0, 0);  // arbitrary
    rec.front_face = true;     // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}
