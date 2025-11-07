cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet_cpp_cuda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(resnet.out resnet.cpp)
target_link_libraries(resnet.out "${TORCH_LIBRARIES}")
set_property(TARGET resnet.out PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET resnet.out
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:resnet.out>)
endif (MSVC)
