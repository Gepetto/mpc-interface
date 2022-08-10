#
# Copyright (c) 2015-2022 CNRS
#

CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(PROJECT_NAME mpc-interface)
SET(PROJECT_DESCRIPTION "A fast and flexible formulation of QP Probl")
SET(PROJECT_URL "http://github.com/gepetto/${PROJECT_NAME}")
SET(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
SET(PROJECT_USE_CMAKE_EXPORT TRUE)
SET(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE True)

# --- OPTIONS  ---------------------------------------
# Need to be set before including base.cmake
# ----------------------------------------------------
OPTION(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
IF(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  IF(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    MESSAGE(FATAL_ERROR "\nPlease run the following command first:\ngit submodule update --init\n")
  ELSE()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  ENDIF()
ENDIF()

# Documentation
SET(DOXYGEN_USE_MATHJAX YES)
SET(DOXYGEN_USE_TEMPLATE_CSS YES)

# Create Project
INCLUDE(${JRL_CMAKE_MODULES}/base.cmake)
COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})

INCLUDE(${JRL_CMAKE_MODULES}/boost.cmake)
INCLUDE(${JRL_CMAKE_MODULES}/python.cmake)
INCLUDE(${JRL_CMAKE_MODULES}/ide.cmake)
INCLUDE(${JRL_CMAKE_MODULES}/apple.cmake)

# Tensor extension imposes CXX 11
# Tensor_to_Matrix imposes CXX 14
CHECK_MINIMAL_CXX_STANDARD(14 ENFORCE)

# Specifies dependencies

ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
IF(BUILD_TESTING)
  FIND_PACKAGE(Boost REQUIRED COMPONENTS unit_test_framework)
ENDIF()

# C++ part, work in progress
add_subdirectory(cpp)
