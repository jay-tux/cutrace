//
// Created by jay on 11/18/23.
//

#include "gpu_types.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <stdio.h>

using namespace cutrace::gpu;

__device__ constexpr bool is_valid(float min, float v) {
  return isfinite(v) && min <= v;
}

struct matrix {
  vector columns[3];

  __device__ float determinant() {
    float a = columns[0].x, b = columns[1].x, c = columns[2].x,
          d = columns[0].y, e = columns[1].y, f = columns[2].y,
          g = columns[0].z, h = columns[1].z, i = columns[2].z;

    return a*e*i + b*f*g + c*d*h - c*e*g - a*f*h - b*d*i;
  }
};

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
    *normal = (p2 - p3).cross(p1 - p3).normalized();
    return true;
  }

  return false;
}

__device__ bool triangle_set::intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const {
  vector h{};
  float t;
  *dist = INFINITY;
  vector n{};
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

__device__ bool sphere::intersect(const cutrace::gpu::ray *r, float min_t, cutrace::gpu::vector *hit, float *dist,
                                  cutrace::gpu::vector *normal) const {
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

__device__ void sun::direction_to(const cutrace::gpu::vector *point, cutrace::gpu::vector *d, float *distance) const {
  *d = -1.0f * direction;
  *distance = INFINITY;
}

__device__ void point_light::direction_to(const cutrace::gpu::vector *p, cutrace::gpu::vector *direction,
                                          float *distance) const {
  *direction = point - p->normalized();
  *distance = (point - *p).norm();
}

