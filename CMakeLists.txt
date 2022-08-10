#
# Copyright (c) 2015-2022 CNRS
#

cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME mpc-interface)
set(PROJECT_DESCRIPTION "A fast and flexible formulation of QP Probl")
set(PROJECT_URL "http://github.com/gepetto/${PROJECT_NAME}")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})

# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nPlease run the following command first:\ngit submodule update --init\n"
    )
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# Documentation
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

# Create Project
include(${JRL_CMAKE_MODULES}/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include(${JRL_CMAKE_MODULES}/boost.cmake)
include(${JRL_CMAKE_MODULES}/python.cmake)
include(${JRL_CMAKE_MODULES}/ide.cmake)
include(${JRL_CMAKE_MODULES}/apple.cmake)

# Tensor extension imposes CXX 11 Tensor_to_Matrix imposes CXX 14
check_minimal_cxx_standard(14 ENFORCE)

# Specifies dependencies

add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
if(BUILD_TESTING)
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif()

# C++ part, work in progress
add_subdirectory(cpp)

# Python part, main code
add_subdirectory(python)

install(FILES package.xml DESTINATION share/${PROJECT_NAME})
