cmake_minimum_required(VERSION 3.16)
project(mio_tts_cpp LANGUAGES C CXX)

option(MIO_TTS_BUILD_SERVER "Build mio-tts-server" ON)
option(MIO_TTS_BUILD_LLAMA_SERVER "Build llama.cpp llama-server target" ON)

set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Path to llama.cpp source tree")
if(NOT LLAMA_CPP_SOURCE_DIR)
    if(DEFINED ENV{LLAMA_CPP_SOURCE_DIR} AND NOT "$ENV{LLAMA_CPP_SOURCE_DIR}" STREQUAL "")
        set(LLAMA_CPP_SOURCE_DIR "$ENV{LLAMA_CPP_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "Set -DLLAMA_CPP_SOURCE_DIR=/path/to/llama.cpp (or env LLAMA_CPP_SOURCE_DIR)")
    endif()
endif()

if(NOT EXISTS "${LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Invalid LLAMA_CPP_SOURCE_DIR: ${LLAMA_CPP_SOURCE_DIR}")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MIO_TTS_BUILD_LLAMA_SERVER)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
else()
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
endif()

# Auto-patch llama.cpp getrows.cu for ROCm APU host-pointer safety if needed
set(GETROWS_CU "${LLAMA_CPP_SOURCE_DIR}/ggml/src/ggml-cuda/getrows.cu")
if(EXISTS "${GETROWS_CU}")
    file(READ "${GETROWS_CU}" GETROWS_SRC)
    if(NOT "${GETROWS_SRC}" MATCHES "src1_dev")
        string(REPLACE
            "get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,"
            "const int32_t * src1_d = (const int32_t *) src1->data;\n    ggml_cuda_pool_alloc<int32_t> src1_dev(ctx.pool());\n    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {\n        src1_dev.alloc(ggml_nelements(src1));\n        CUDA_CHECK(cudaMemcpyAsync(src1_dev.ptr, src1->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, stream));\n        src1_d = src1_dev.ptr;\n    }\n    get_rows_cuda(src0->data, src0->type, src1_d, dst->data, dst->type,"
            GETROWS_PATCHED
            "${GETROWS_SRC}"
        )
        file(WRITE "${GETROWS_CU}" "${GETROWS_PATCHED}")
        message(STATUS "Patched ${GETROWS_CU} for ROCm APU host pointer safety")
    endif()
endif()

add_subdirectory(${LLAMA_CPP_SOURCE_DIR} ${CMAKE_BINARY_DIR}/llama_cpp EXCLUDE_FROM_ALL)

if(NOT TARGET cpp-httplib)
    add_subdirectory(${LLAMA_CPP_SOURCE_DIR}/vendor/cpp-httplib ${CMAKE_BINARY_DIR}/cpp_httplib EXCLUDE_FROM_ALL)
endif()

find_package(Threads REQUIRED)

add_library(mio-tts-lib STATIC
    src/mio-tts-lib.cpp
    src/miocodec-decoder.cpp
    src/wavlm-extractor.cpp
)
target_compile_features(mio-tts-lib PRIVATE cxx_std_17)
target_include_directories(mio-tts-lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE
        ${LLAMA_CPP_SOURCE_DIR}/vendor
)
target_link_libraries(mio-tts-lib
    PUBLIC
        llama
        Threads::Threads
)

add_executable(llama-tts-mio src/tts-mio-cli.cpp)
target_compile_features(llama-tts-mio PRIVATE cxx_std_17)
target_include_directories(llama-tts-mio PRIVATE ${LLAMA_CPP_SOURCE_DIR} ${LLAMA_CPP_SOURCE_DIR}/vendor)
target_link_libraries(llama-tts-mio PRIVATE mio-tts-lib llama Threads::Threads cpp-httplib)

if(MIO_TTS_BUILD_SERVER)
    add_executable(mio-tts-server src/tts-mio-server.cpp)
    target_compile_features(mio-tts-server PRIVATE cxx_std_17)
    target_include_directories(mio-tts-server PRIVATE ${LLAMA_CPP_SOURCE_DIR} ${LLAMA_CPP_SOURCE_DIR}/vendor)
    target_link_libraries(mio-tts-server PRIVATE mio-tts-lib llama Threads::Threads)
    target_link_libraries(mio-tts-server PRIVATE cpp-httplib)
endif()
