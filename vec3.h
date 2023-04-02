#pragma once

#include "rtweekend.h"

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {
public:
	__host__ __device__ vec3() : e{ 0,0,0 } {}
	__host__ __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}
	// vec3(vec3& vector) : e{ *vector.e } {}

	// Get coordinate values
	__host__ __device__ double x() const { return e[0]; }
	__host__ __device__ double y() const { return e[1]; }
	__host__ __device__ double z() const { return e[2]; }

	// Get negative vector
	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	
	// Access values by index
	__host__ __device__ double operator[](int i) const { return e[i]; }
	__host__ __device__ double& operator[](int i) { return e[i]; }

	// Vector addition and subtraction
	__host__ __device__
	vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__
	vec3& operator-=(const vec3& v) {
		return *this += (-v);
	}

	// Scalar multiplication and division
	__host__ __device__
	vec3& operator*=(const double t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	__host__ __device__
	vec3& operator/=(const double t) {
		return *this *= 1 / t;
	}

	// Magnitude
	__host__ __device__
	double length() const {
		return sqrt(length_squared());
	}

	__host__ __device__
	double length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	
	// Create random vectors
	__device__
	inline static vec3 random(curandState* local_rand_state) {
		return vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
	}

	__device__
	inline static vec3 random(double min, double max, curandState* local_rand_state) {
		double difference = max - min;
		return vec3((curand_uniform(local_rand_state)*difference) + min,
			(curand_uniform(local_rand_state) * difference) + min,
			(curand_uniform(local_rand_state) * difference) + min);
	}

	// Avoid unwanted 0 vectors
	__host__ __device__
	bool near_zero() const {
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}

	double e[3];
};

//Type aliases
using point3 = vec3;
using color = vec3;

// Utility functions

// Printing vectors
__host__ __device__
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

// Vector addition and subtraction
__host__ __device__
inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__
inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// Scalar multiplication and division
__host__ __device__
inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__
inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

__host__ __device__
inline vec3 operator/(vec3 v, double t) {
	return (1 / t) * v;
}

// Vector multiplication
// Hadamard Product 
__host__ __device__
inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

// Dot product
__host__ __device__
inline double dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

// Cross product
__host__ __device__
inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

// Unit vector
__host__ __device__
inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}


// Random vectors
 __device__
vec3 random_in_unit_sphere(curandState* local_rand_state) {
	while (true) {
		auto p = vec3::random(-1, 1, local_rand_state);
		if (p.length_squared() < 1) return p;
	}
}

__device__
vec3 random_unit_vector(curandState* local_rand_state) {
	return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__
vec3 random_in_unit_disk(curandState *local_rand_state) {
	while (true) {
		vec3 p = vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0);
		if (p.length_squared() < 1) return p;
	}
}

// Vector math for pure reflection
__host__ __device__
vec3 reflect(const vec3& v, const vec3& n) {
	// Scale the unit vector n to 2*dot product
	return v - (2 * dot(v, n) * n);
}

// Vector math for arbitrary refraction
__host__ __device__
vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
	auto cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_parallel + r_out_perp;
}