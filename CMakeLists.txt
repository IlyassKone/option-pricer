cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_core cpp/src/mc_pricer.cpp cpp/src/bindings.cpp)
target_include_directories(_core PRIVATE cpp/include)
target_compile_options(_core PRIVATE -Wall -Wextra)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Installs _core next to the Python files: option_pricer/_core.*.so
install(TARGETS _core LIBRARY DESTINATION option_pricer)
