cmake_minimum_required(VERSION 3.20)

project(print_latex CXX)

find_package(Torch REQUIRED)
message(STATUS "Torch version ${Torch_VERSION}")

#message(" ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(${PROJECT_NAME} main.cpp example1.hpp example2.hpp example3.hpp example4.hpp)
add_library(
        ${PROJECT_NAME}_lib
        SHARED
        positional_encoding.hpp
        resnet_transformer.hpp
        tokenizer.hpp
        utils.hpp
        predict.hpp
        predict.cpp
)
set_target_properties(
        ${PROJECT_NAME}_lib
        PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/out/debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/out/release
)
target_link_libraries(
        ${PROJECT_NAME}_lib
        PUBLIC
        "${TORCH_LIBRARIES}"
        "resnet"
)

add_executable(
        ${PROJECT_NAME}
        main.cpp
        opencv_image_io.hpp
        im2latex.hpp
        data_loader.hpp
        train.hpp
)

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

set(OpenCV_DIR "C:/Users/15258/work/opencv/build")
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


message("${TORCH_LIBRARIES} ${STB_LIB_NAME}")
target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/out/release)
target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC
        ${PROJECT_NAME}_lib
        ${OpenCV_LIBS}
)

if (MSVC)
    include(cmake/copy_torch_dlls.cmake)
    copy_torch_dlls(${PROJECT_NAME})
endif ()

#add_subdirectory(tests)