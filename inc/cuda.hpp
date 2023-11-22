//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_CUDA_HPP
#define CUTRACE_CUDA_HPP

#include <iostream>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define cudaCheck(call)                                                        \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess)                                                    \
      std::cerr << "[CUDA] Error " << (size_t)err << "("                       \
                << cudaGetErrorName(err) << ") in " << __FILE__                \
                << ", on line " << __LINE__ << std::endl                       \
                << "   -> while executing " << #call << std::endl              \
                << "   -> Error message: " << cudaGetErrorString(err)          \
                << std::endl;                                                  \
  }


namespace cutrace {
template <typename T>
inline T checked(T t, const char *call, const char *file, int line) {
  cudaError_t err = cudaPeekAtLastError();
  if(err != cudaSuccess) {
    std::cerr << "[CUDA] Error " << (size_t)err << "(" << cudaGetErrorName(err)
              << ") in " << __FILE__ << ", on line " << __LINE__ << "\n"
              << "   -> while executing " << call << "\n" << "   -> Error message: "
              << cudaGetErrorString(err) << "\n";
  }
  return t;
}
}

#define cudaChecked(call) cutrace::checked(call, #call, __FILE__, __LINE__);

#endif //CUTRACE_CUDA_HPP
