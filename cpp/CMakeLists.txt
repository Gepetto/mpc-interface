# Create target libmpc-interface.so

# Specify sources
set(${PROJECT_NAME}_HEADERS include/${PROJECT_NAME}/dynamics.hh
                            include/${PROJECT_NAME}/tools.hh)
set(${PROJECT_NAME}_SOURCES src/dynamics.cc src/tools.cc)

# Create library
add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_HEADERS}
                                   ${${PROJECT_NAME}_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)

# Install procedure
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
