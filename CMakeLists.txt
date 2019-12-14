cmake_minimum_required( VERSION 2.6 )
project( opencvDnn )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")
set_property(GLOBAL PROPERTY USE_FOLDERS On)

set(OpenCV_DIR /opt/intel/openvino/opencv/cmake/)
find_package(OpenCV REQUIRED)
include_directories( ${OPENCV_INCLUDE_DIR})

link_directories(/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/)

add_executable(faceDetector faceDetector.cpp)
target_link_libraries(faceDetector ${OpenCV_LIBS} inference_engine) 

#add_executable(tsaiCalibration tsaiCalibration.cpp)
#target_link_libraries(tsaiCalibration ${OpenCV_LIBS} ) 

#add_executable(zzhangCalibration zzhangCalibration.cpp dltalgo.cpp parser.cpp)
#target_link_libraries(zzhangCalibration ${OpenCV_LIBS} ) 
