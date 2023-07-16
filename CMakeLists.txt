cmake_minimum_required(VERSION 3.20)

project(print_latex CXX)

find_package(Torch REQUIRED)
message(STATUS "Torch version ${Torch_VERSION}")

#message(" ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(${PROJECT_NAME} main.cpp example1.hpp example2.hpp example3.hpp example4.hpp)

add_executable(
        ${PROJECT_NAME}
        main.cpp
        data_loader.hpp
#        stb_image_io.hpp
        train.hpp
        positional_encoding.hpp
        resnet_transformer.hpp
        tokenizer.hpp
        im2latex.hpp
        utils.hpp
        opencv_image_io.hpp)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/out/debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/out/release
)
## resnet begin
add_subdirectory(resnet)
## resnet end

## include stb
#include(FetchContent)
#set(STB_LIB_NAME stb)
#FetchContent_Declare(
#        ${STB_LIB_NAME}
#        GIT_REPOSITORY https://github.com/nothings/stb.git
#        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb
#)
#FetchContent_MakeAvailable(${STB_LIB_NAME})
#add_library(${STB_LIB_NAME} INTERFACE)
#target_sources(
#        ${STB_LIB_NAME}
#        INTERFACE
#        ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb/stb_image.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb/stb_image_write.h
#        ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb/stb_image_resize.h
#)
#target_include_directories(${STB_LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb)

set(OpenCV_DIR "C:/Users/15258/work/opencv/build")
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


message("${TORCH_LIBRARIES} ${STB_LIB_NAME}")
target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/out/release)
target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        "${TORCH_LIBRARIES}"
#        ${STB_LIB_NAME}
        "resnet"
        ${OpenCV_LIBS}
)

if (MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif ()

add_subdirectory(tests)