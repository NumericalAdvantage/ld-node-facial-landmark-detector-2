# This file is part of project link.developers/ld-node-facial-landmark-detector-2.
# It is copyrighted by the contributors recorded in the version control history of the file,
# available from its original location https://gitlab.com/link.developers.beta/ld-node-facial-landmark-detector-2.
#
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.0)
project(ld-node-facial-landmark-detector-2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--unresolved-symbols=ignore-in-shared-libs")
endif()

find_package(Link2-Cpp REQUIRED)
find_package(Flatc REQUIRED)
find_package(ld-cmake-tools REQUIRED)
find_package(ld-lib-image REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorflowCC COMPONENTS Shared REQUIRED)

if(TensorflowCC_Shared_FOUND)
	get_property(TF_LIBS TARGET TensorflowCC::Shared PROPERTY INTERFACE_LINK_LIBRARIES)
	get_property(TF_INCLUDE_DIRS TARGET TensorflowCC::Shared PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
	message("Tensorflow Libs: ${TF_LIBS}")
	message("Tensorflow Headers: ${TF_INCLUDE_DIRS}")
else()
	message("Tensorflow library NOT FOUND")
endif()

find_package(CUDA)
if (CUDA_FOUND)
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)

if (WIN32)
add_definitions(-DCOMPILER_MSVC)
endif ()

add_definitions(-DNOMINMAX)


# When adding new data formats, add them to the INPUT_FILES section.
# If you use libraries that contain Flatbuffer files, uncomment the lines with LD_FLATBUFFER_ variables.
flatc_generate_cpp(
    BFBS_OUTPUT_FOLDER ${CMAKE_SOURCE_DIR}/data
    INCLUDE_PATHS ${LD_FLATBUFFER_DATA_FOLDERS} 
    INPUT_FILES
        ${LD_FLATBUFFER_FILES}
        data/ImageWithBB.fbs
    )

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/FacialLandmarkDetector.cpp
    src/FacialLandmarkDetector.h
    src/LandmarkPredictor.h
    src/LandmarkPredictor.cpp
    src/FaceTransform.h
    src/FaceTransform.cpp
    ${FLATC_GENERATED_SOURCES}
    )

target_include_directories(${PROJECT_NAME} PRIVATE ${FLATC_GENERATED_INCLUDE_DIRS} PUBLIC ${OpenCV_INCLUDE_DIRS} ${TF_INCLUDE_DIRS}) 
target_link_libraries(${PROJECT_NAME} PUBLIC DRAIVE::Link2-Cpp ${ld-node-tensorflow-poc_LIBRARIES}  ${TF_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})

cmake_make_installation(
        TARGETS ${PROJECT_NAME}
        LINK2_STATIC_ASSETS
            specification.json
            data
        IS_EXECUTABLE)
