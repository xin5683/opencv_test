cmake_minimum_required(VERSION 3.5)

include_directories(/home/ft/test/tensorflow
                    /home/ft/test/tensorflow/tensorflow/lite/tools/make/downloads/absl
                    /home/ft/test/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)


project(opencv_test)

add_executable(${PROJECT_NAME} main.cpp)

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)



# Link your application with OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS}
                    Threads::Threads
                    /home/ft/test/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a
                    /home/ft/test/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/benchmark-lib.a)

