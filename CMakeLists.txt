cmake_minimum_required(VERSION 3.10)

project(training_data_loader LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(training_data_loader SHARED training_data_loader.cpp)

if(MSVC)
  target_compile_options(training_data_loader PRIVATE
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:RelWithDebInfo>:/O2>)
else()
  target_compile_options(training_data_loader PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-g>
    $<$<CONFIG:RelWithDebInfo>:-O3>)
endif()

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)

include(CTest)
if(BUILD_TESTING)
  add_executable(training_data_loader_tests tests/native_loader_tests.cpp)
  target_link_libraries(training_data_loader_tests Threads::Threads)
  if(MSVC)
    target_compile_options(training_data_loader_tests PRIVATE $<$<CONFIG:Release>:/O2>)
  else()
    target_compile_options(training_data_loader_tests PRIVATE $<$<CONFIG:Release>:-O3>)
  endif()
  add_test(NAME training_data_loader_tests COMMAND training_data_loader_tests)
endif()

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
