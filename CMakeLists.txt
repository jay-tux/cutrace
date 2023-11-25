cmake_minimum_required(VERSION 3.26)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_COMPILER_ID "gcc")
set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES conan_provider.cmake)
project(cutrace CUDA)

include(FetchContent)
FetchContent_Declare(
        picojson
        GIT_REPOSITORY https://github.com/kazuho/picojson.git
)
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
)

add_executable(cutrace
        main.cu
        src/loader.cu
        src/gpu_types.cu
        src/cpu_types.cu
        src/kernel_depth.cu
        src/images.cu
        src/shading.cu
        src/mesh_loader.cu
        src/kernel_prepare.cu
)

#conan_install()

find_package(ASSIMP REQUIRED)

set_target_properties(cutrace PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
FetchContent_MakeAvailable(picojson stb)
target_include_directories(${PROJECT_NAME} PUBLIC "${picojson_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PUBLIC "${stb_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/inc")
target_link_libraries(${PROJECT_NAME} assimp::assimp)