macro(ADD_QP_FORMULATIONS_UNIT_TEST NAME)
  set(TEST_NAME "test-cpp-${NAME}")
  add_unit_test(${TEST_NAME} ${NAME}.cpp)
  target_link_libraries(${TEST_NAME} PRIVATE ${PROJECT_NAME}
                                             Boost::unit_test_framework)
  target_compile_options(
    ${TEST_NAME} PRIVATE "-DBOOST_TEST_DYN_LINK"
                         "-DBOOST_TEST_MODULE=${MODULE_NAME}")
endmacro(ADD_QP_FORMULATIONS_UNIT_TEST)

add_qp_formulations_unit_test(tools)
