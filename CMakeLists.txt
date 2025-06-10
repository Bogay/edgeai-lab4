cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(MyWhisperApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    whisper_cpp
    GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
    GIT_TAG        v1.7.5 
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(whisper_cpp)

add_executable(my_whisper_app src/main.cpp src/common-whisper.cpp src/common.cpp)

target_link_libraries(my_whisper_app PRIVATE whisper)

# (Optional) If you want to enable specific whisper.cpp build options
# For example, to enable BLAS for potentially better CPU performance on RPi
# set(WHISPER_USE_BLAS ON CACHE BOOL "" FORCE)
# set(WHISPER_OPENBLAS ON CACHE BOOL "" FORCE) # Or WHISPER_CLBLAST, WHISPER_CUBLAS etc.
# For Raspberry Pi, using `WHISPER_OPENBLAS` if you have `libopenblas-dev` installed
# might give a speedup. Check whisper.cpp's documentation for optimal BLAS setup.
