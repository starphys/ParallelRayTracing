#pragma once

#include "vec3.h"

class ray {
public:
	__device__ ray() {}
	__device__ ray(const point3& origin, const vec3& direction, double time = 0.0)
		: orig(origin), dir(direction), tm(time) {}

	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }
	__device__ double time() const { return tm; }

	__device__
	point3 at(double t) const {
		return orig + t * dir;
	}

	point3 orig;
	vec3 dir;
	double tm;
};