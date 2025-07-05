cmake_minimum_required(VERSION 3.10)
project(MLP)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(../mlp)
aux_source_directory(../mlp SRC_LIST)
aux_source_directory(./scr SRC_LIST)
add_executable(demo ${SRC_LIST})
target_link_libraries(demo m)
