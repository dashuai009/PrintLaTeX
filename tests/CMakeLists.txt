
add_executable(test_opencv test_opencv.cpp ../opencv_image_io.hpp)
target_link_libraries(test_opencv PUBLIC
        "${TORCH_LIBRARIES}"
        ${OpenCV_LIBS})
set_target_properties(test_opencv
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../out/debug
        RUNTIME_OUTPUT_DIRECTORY_RE ${CMAKE_CURRENT_SOURCE_DIR}/../out/debug
)
message(STATUS ${OPENCV_INSTALL_PREFIX})
#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#    add_custom_command(TARGET ${TARGET_NAME}
#            POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_if_different
#            ${TORCH_DLLS}
#            $<TARGET_FILE_DIR:${TARGET_NAME}>
#            )
#endif (MSVC)
add_executable(test_rand test_rand.cpp)

add_executable(test_in test_in.cpp)
target_link_libraries(test_in PUBLIC
        "resnet"
        "${TORCH_LIBRARIES}"
        ${OpenCV_LIBS})
target_include_directories(test_in PRIVATE ${torch_INCLUDE_DIRS})
set_target_properties(test_in
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../out/debug
        RUNTIME_OUTPUT_DIRECTORY_RE ${CMAKE_CURRENT_SOURCE_DIR}/../out/debug
        )

message(STATUS "include path:${CMAKE_INCLUDE_PATH}")