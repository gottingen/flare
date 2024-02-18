add_library(af_glad_obj_lib OBJECT ${CMAKE_CURRENT_LIST_DIR}/src/glad.c)

set_target_properties(af_glad_obj_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(af_glad_obj_lib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
