cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

option(ENABLE_HIP  "Enable HIP (AMD GPU) support"     OFF)
option(ENABLE_CUDA "Enable CUDA (NVIDIA GPU) support" OFF)

# ---- project + language enablement -------------------------------------
if(ENABLE_HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
  endif()
  project(CompressMGARDCentroidOperator C CXX HIP)
  find_package(hip REQUIRED)
  set(GPU_TARGETS "gfx90a" CACHE STRING "GPU architectures")
  message(STATUS "HIP enabled: ROCm path = ${ROCM_PATH}")
elseif(ENABLE_CUDA)
  project(CompressMGARDCentroidOperator C CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
else()
  project(CompressMGARDCentroidOperator C CXX)
  message(STATUS "Building CPU-only version (no GPU acceleration)")
endif()

#------------------------------------------------------------------------------#
# Some boilerplate to setup nice output directories
#------------------------------------------------------------------------------#
include(GNUInstallDirs)

find_package(ADIOS2 REQUIRED)
if(ENABLE_HIP OR ENABLE_CUDA)
  find_package(MGARD REQUIRED)
endif()
set(BUILD_SHARED_LIBS ON)

# ---- imported serial MGARD (CPU-only builds) ---------------------------
set(MGARD_SERIAL_DIR
    "/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD-serial/install-serial"
    CACHE PATH "Path to serial-only MGARD installation")
add_library(mgard_serial SHARED IMPORTED GLOBAL)
set_target_properties(mgard_serial PROPERTIES
  IMPORTED_LOCATION             "${MGARD_SERIAL_DIR}/lib64/libmgard.so"
  INTERFACE_INCLUDE_DIRECTORIES "${MGARD_SERIAL_DIR}/include"
  INTERFACE_COMPILE_DEFINITIONS "MGARD_ENABLE_SERIAL=1;MGARD_ZSTD")

# ------------------------------------------------------------------------
# MGARD-X Huffman+ZSTD lossless backends.
# ------------------------------------------------------------------------
if(ENABLE_HIP OR ENABLE_CUDA)
  set(_MGARDX_LOSSLESS_SERIAL_MGARD mgard::mgard)
else()
  set(_MGARDX_LOSSLESS_SERIAL_MGARD mgard_serial)
endif()

add_library(mgardx_lossless_serial STATIC
            MGARDXLossless_SERIAL.cpp MGARDXLossless.h)
set_source_files_properties(MGARDXLossless_SERIAL.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(mgardx_lossless_serial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
# Uses Huffman+Zstd (ComposedLosslessCompressor<SERIAL>) when max_code <
# dict_size, raw Zstd otherwise.  outlier_count uninitialized bug patched
# in-place without modifying MGARD-X headers.
set(ZSTD_LIBRARY "${MGARD_SERIAL_DIR}/lib/libzstd.so"
    CACHE FILEPATH "Path to libzstd.so used by mgardx_lossless_serial")

target_include_directories(mgardx_lossless_serial PRIVATE
  "${MGARD_SERIAL_DIR}/include")
target_link_libraries(mgardx_lossless_serial
  PRIVATE ${_MGARDX_LOSSLESS_SERIAL_MGARD}
          "${ZSTD_LIBRARY}")
set_target_properties(mgardx_lossless_serial PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

if(ENABLE_HIP)
  message(STATUS "HIP enabled ==> build mgardx_lossless_hip")
  add_library(mgardx_lossless_hip STATIC MGARDXLossless_HIP.cpp MGARDXLossless.h)
  set_source_files_properties(MGARDXLossless_HIP.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(mgardx_lossless_hip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_link_libraries(mgardx_lossless_hip
    PRIVATE hip::device mgard::mgard "${ZSTD_LIBRARY}")
  target_compile_definitions(mgardx_lossless_hip PRIVATE
    __HIP_PLATFORM_AMD__ ENABLE_HIP MGARDX_COMPILE_HIP)
  set_target_properties(mgardx_lossless_hip PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
    POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------------
# Device kernel libraries (centroid split / SFC / quantize).
# ------------------------------------------------------------------------
add_library(device_kernels_cpu STATIC
            DeviceKernels_CPU.cpp DeviceKernels.h)
set_source_files_properties(DeviceKernels_CPU.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(device_kernels_cpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
set_target_properties(device_kernels_cpu PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
if(ENABLE_HIP)
  target_compile_definitions(device_kernels_cpu PUBLIC CMG_HAVE_HIP)
endif()

if(ENABLE_HIP)
  message(STATUS "HIP enabled ==> build device_kernels_hip")
  add_library(device_kernels_hip STATIC
              DeviceKernels_HIP.cpp DeviceKernels.h)
  set_source_files_properties(DeviceKernels_HIP.cpp PROPERTIES LANGUAGE HIP)
  target_include_directories(device_kernels_hip PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_link_libraries(device_kernels_hip PRIVATE hip::device)
  target_compile_definitions(device_kernels_hip PRIVATE
    __HIP_PLATFORM_AMD__ ENABLE_HIP CMG_HAVE_HIP)
  set_target_properties(device_kernels_hip PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
    POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------------
# Unified centroid operator (single .so).
# ------------------------------------------------------------------------
add_library(CompressMGARDCentroidOperator
            CompressMGARDCentroidOperator.cpp
            CompressMGARDCentroidOperator.h)

# adios2::cxx is the public C++ API used for the mesh reads; the v2 plugin
# interface itself is header-only, so no link against internal adios2 targets.
target_link_libraries(CompressMGARDCentroidOperator
  PRIVATE device_kernels_cpu mgardx_lossless_serial
  PUBLIC  adios2::cxx)

if(ENABLE_HIP)
  message(STATUS "HIP enabled ==> build CompressMGARDCentroidOperator with HIP")
  set_source_files_properties(CompressMGARDCentroidOperator.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(CompressMGARDCentroidOperator
    PRIVATE device_kernels_hip mgardx_lossless_hip hip::device
    PUBLIC  mgard::mgard)
  target_compile_definitions(CompressMGARDCentroidOperator PRIVATE
    __HIP_PLATFORM_AMD__ ENABLE_HIP CMG_HAVE_HIP)
else()
  message(STATUS "CPU only ==> build CompressMGARDCentroidOperator CPU only")
  set_source_files_properties(CompressMGARDCentroidOperator.cpp PROPERTIES LANGUAGE CXX)
  target_link_libraries(CompressMGARDCentroidOperator
    PUBLIC mgard_serial)
endif()

include(GenerateExportHeader)
generate_export_header(CompressMGARDCentroidOperator
                       BASE_NAME compress_mgard_centroid)

target_include_directories(CompressMGARDCentroidOperator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

install(TARGETS CompressMGARDCentroidOperator LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

message(STATUS "=== CompressMGARDCentroidOperator (unified V5) ===")
message(STATUS "  HIP support:  ${ENABLE_HIP}")
message(STATUS "  ADIOS2:       ${ADIOS2_DIR}")
message(STATUS "  MGARD:        ${MGARD_DIR}")
message(STATUS "  Target:       libCompressMGARDCentroidOperator.so")
message(STATUS "  Dispatch:     CENTROID_DEVICE={cpu,gpu} (default cpu)")
message(STATUS "===================================================")
