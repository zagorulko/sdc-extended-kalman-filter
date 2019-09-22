project(ExtendedKF)

cmake_minimum_required(VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources
  src/main.cpp
  src/fusion_ekf.cpp
  src/kalman_filter.cpp
  src/tools.cpp
)

include_directories(external)

include_directories(/usr/local/include)
link_directories(/usr/local/lib64)
link_directories(/usr/local/lib)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  include_directories(/usr/local/opt/openssl/include)
  link_directories(/usr/local/opt/openssl/lib)
  link_directories(/usr/local/Cellar/libuv/1*/lib)
endif()

add_executable(ExtendedKF ${sources})
target_link_libraries(ExtendedKF z ssl uv uWS)
