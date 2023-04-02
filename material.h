#pragma once

#include "rtweekend.h"
#include "texture.h"

struct hit_record;

class material {
public:
	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const = 0;

	__device__
	virtual color emitted(double u, double v, const point3& p) const {
		return color(0, 0, 0);
	}
};

class lambertian : public material {
public:
	__device__ lambertian(const color& a) : albedo(new solid_color(a)) {}
	__device__ lambertian(texture* a) : albedo(a) {}

	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const override {
		auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);
		
		// No zero/near zero vectors!
		if (scatter_direction.near_zero()) {
			scatter_direction = rec.normal;
		}
		
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	texture* albedo;
};

class metal : public material {
public:
	__device__ metal(const color& a, double f) : albedo(new solid_color(a)), fuzz(f < 1 ? f : 1) {}
	__device__ metal(texture* a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	texture* albedo;
	double fuzz;
};

class dielectric : public material {
public:
	__device__ dielectric(double index_of_refraction, const color& a = color(1.0,1.0,1.0)) : ir(index_of_refraction), albedo(new solid_color(a)) {}

	__device__ dielectric(double index_of_refraction, texture* a) : ir(index_of_refraction), albedo(a) {}

	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const override {
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}

	double ir;
	texture* albedo;

private:
	__device__
	static double reflectance(double cosine, double ref_idx) {
		// Schlick's approximation
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};

class diffuse_light : public material {
public:
	__device__ diffuse_light(texture* a) : emit(a) {}
	__device__ diffuse_light(color c) : emit(new solid_color(c)) {}

	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const override {
		return false;
	}

	__device__
	virtual color emitted(double u, double v, const point3& p) const override {
		return emit->value(u, v, p);
	}

	texture* emit;
};


class isotropic : public material {
public:
	__device__ isotropic(color c) : albedo(new solid_color(c)) {}
	__device__ isotropic(texture* a) : albedo(a) {}

	__device__
	virtual bool scatter(
		const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state
	) const override {
		scattered = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

	texture* albedo;
};