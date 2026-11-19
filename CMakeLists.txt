# CMakeList.txt: light_field_tracking 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.20)

project ("light_field_tracking" VERSION 1.0.0 LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(OpenCV_DIR "D:/opencv/build/x64/vc16/lib")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pylon CONFIG REQUIRED)
# 将源代码添加到此项目的可执行文件。
add_executable (light_field_tracking "main.cpp" "include/light_field_tracking.h" "cuda/refocus.cu" "include/Icamera.h" "sorting.cpp" "globals.cpp" "Camera.cpp")



# TODO: 如有需要，请添加测试并安装目标。
target_include_directories(light_field_tracking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDAToolkit_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(light_field_tracking PRIVATE
    CUDA::cudart
    PRIVATE pylon::pylon 
    ${OpenCV_LIBS}
)
set_target_properties(light_field_tracking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(
    light_field_tracking PRIVATE
    PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)