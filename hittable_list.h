#pragma once

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list : public hittable {
public:
	__device__ hittable_list() {}
	__device__ hittable_list(hittable** l, int n) { list = l; list_size = n; }

	__device__
	virtual bool hit(
		const ray& r, double t_min, double t_max, hit_record& rec) const override;

	__device__
	virtual bool bounding_box(
		double time0, double time1, aabb& output_box) const override;

	hittable **list;
	int list_size;
};

__device__
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (int i = 0; i < list_size; ++i) {
		if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

__device__
bool hittable_list::bounding_box(double time0, double time1, aabb& output_box) const {
	if (!list || list_size == 0) return false;

	aabb temp_box;
	bool first_box = true;

	for (int i = 0; i < list_size; ++i) {
		if (!list[i]->bounding_box(time0, time1, temp_box)) return false;
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
	}

	return true;
}