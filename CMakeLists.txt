cmake_minimum_required(VERSION 3.1)

project(sc_aloam)

set(CMAKE_CXX_STANDARD 14)# for the problem of error: ‘make_unique’ is not a member of ‘std’
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(PCL REQUIRED)
include_directories(
                    ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS}
                    ${GTSAM_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/include   
                    ${PROJECT_SOURCE_DIR}/Scancontext
)           

find_package(
    catkin REQUIRED COMPONENTS
        roscpp
        tf
        sensor_msgs
        pcl_conversions
        nav_msgs
        geometry_msgs
        cv_bridge
)
include_directories(${catkin_INCLUDE_DIRS})

catkin_package()

add_executable(loopclosure_aloam src/SC_Optimization_node.cpp
                                 Scancontext/Scancontext.cpp)
target_link_libraries(loopclosure_aloam ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)
