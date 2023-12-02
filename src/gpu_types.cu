//
// Created by jay on 11/18/23.
//

#include "gpu_types.hpp"

using namespace cutrace;
using namespace cutrace::gpu;

__device__ constexpr bool is_valid(float min, float v) {
  return isfinite(v) && min <= v;
}

__device__ bool triangle::intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const {
  auto a = p2 - p1;
  auto b = p2 - p3;
  auto c = r->dir;
  auto d = p2 - r->start;

  matrix A{{a, b, d}};
  matrix B{{a, b, c}};
  matrix A1{{d,b,c}};
  matrix A2{{a,d,c}};

  float beta = A1.determinant() / B.determinant();
  float gamma = A2.determinant() / B.determinant();
  float t0 = A.determinant() / B.determinant();

  if (beta >= 0 && gamma >= 0 && beta + gamma <= 1 && is_valid(min_t, t0)) {
    *dist = t0;
    *hit = r->start + *dist * r->dir;
    *normal = -1.0f * (p2 - p3).cross(p1 - p3).normalized();
    return true;
  }

  return false;
}

__device__ inline bool bound_intersect(const ray *r, const bound *b) {
  // from https://tavianator.com/2022/ray_box_boundary.html
  float tmin = 0.0, tmax = INFINITY;

  vector r_inv = { 1.0f / r->dir.x, 1.0f / r->dir.y, 1.0f / r->dir.z };

  for (int d = 0; d < 3; ++d) {
    float t1 = (b->min[d] - r->start[d]) * r_inv[d];
    float t2 = (b->max[d] - r->start[d]) * r_inv[d];

    tmin = min(max(t1, tmin), max(t2, tmin));
    tmax = max(min(t1, tmax), min(t2, tmax));
  }

  return tmin <= tmax;
}

__device__ bool triangle_set::intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const {
  vector h{};
  float t;
  *dist = INFINITY;
  vector n{};

  // culling
  if(!bound_intersect(r, &bounding_box)) return false;

  for(const auto &tri : triangles) {
    if(tri.intersect(r, min_t, &h, &t, &n) && t < *dist) {
      *dist = t;
      *hit = h;
      *normal = n;
    }
  }

  return *dist != INFINITY;
}

__device__ bool plane::intersect(const ray *r, float min_t, vector *hit, float *dist, vector *n) const {
  float t0 = (point - r->start).dot(normal) / r->dir.dot(normal);

  if(is_valid(min_t, t0)) {
    *dist = t0;
    *hit = r->start + t0 * r->dir;
    *n = this->normal;
    return true;
  }

  return false;
}

__device__ bool sphere::intersect(const cutrace::gpu::ray *r, float min_t, cutrace::vector *hit, float *dist,
                                  cutrace::vector *normal) const {
  auto d = r->dir.normalized();
  auto c = center;
  auto e = r->start;
  auto R = radius;

  auto dec = -d.dot(e - c);
  auto sub = (d.dot(e - c) * d.dot(e - c)) - d.dot(d) * ((e - c).dot(e - c) - R * R);

  auto t0 = (dec - sqrt(sub)) / d.dot(d);
  auto t1 = (dec + sqrt(sub)) / d.dot(d);

  auto is_valid = [min_t](float f) { return isfinite(f) && min_t <= f; };

  if(!is_valid(t0)) {
    if(!is_valid(t1)) return false;
    else *dist = t1;
  }
  else {
    if(!is_valid(t1)) *dist = t0;
    else *dist = min(t0, t1);
  }

  *hit = r->start + *dist * r->dir.normalized();
  *normal = (*hit - c).normalized();
  return true;
}

__device__ void sun::direction_to(const cutrace::vector *, cutrace::vector *d, float *distance) const {
  *d = -1.0f * direction;
  *distance = INFINITY;
}

__device__ void point_light::direction_to(const cutrace::vector *p, cutrace::vector *direction,
                                          float *distance) const {
  *direction = (point - *p).normalized();
  *distance = (point - *p).norm();
}