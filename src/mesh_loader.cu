//
// Created by jay on 11/24/23.
//

#include <iostream>
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "mesh_loader.hpp"

using namespace cutrace;
using namespace cutrace::cpu;

inline vector to_vec(const aiVector3D &v) {
  return {
    .x = v.x,
    .y = v.y,
    .z = v.z
  };
}

inline vector load_pt(const aiMesh *mesh, int idx) {
  return to_vec(mesh->mVertices[idx]);
}

inline triangle load_tri(const aiMesh *mesh, uint *idx) {
  return {
    .p1 = load_pt(mesh, idx[0]),
    .p2 = load_pt(mesh, idx[1]),
    .p3 = load_pt(mesh, idx[2]),
  };
}

std::vector<triangle_set> cpu::load_mesh(const std::string &file, size_t mat_idx) {
  // Create an instance of the Importer class
  Assimp::Importer importer;

  // And have it read the given file with some example postprocessing
  // Usually - if speed is not the most important aspect for you - you'll
  // probably to request more postprocessing than we do in this example.
  const aiScene* scene = importer.ReadFile( file.c_str(),
                                            aiProcess_CalcTangentSpace |
                                            aiProcess_Triangulate             |
                                            aiProcess_JoinIdenticalVertices   |
                                            aiProcess_SortByPType);

  // If the import failed, report it
  if (scene == nullptr) {
    std::cerr << "Loading of " << file << " failed.\n";
    return {};
  }

  std::vector<triangle_set> res;
  res.reserve(scene->mNumMeshes);

  for(size_t mesh_id = 0; mesh_id < scene->mNumMeshes; mesh_id++) {
    const auto *mesh = scene->mMeshes[mesh_id];

    if(mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE)
      continue;

    triangle_set set{};

    for(size_t face_id = 0; face_id < mesh->mNumFaces; face_id++) {
      const auto &face = mesh->mFaces[face_id];
      if(face.mNumIndices != 3)
        continue;

      set.tris.push_back(load_tri(mesh, face.mIndices));
    }

    set.mat_idx = mat_idx;
    res.push_back(set);
  }

  return res;
}