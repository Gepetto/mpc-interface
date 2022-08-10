SET(PYTHON_TESTS
  test_body.py
  test_dynamics.py
  test_goal.py
  test_restrictions.py
  test_tools.py)

FOREACH(test ${PYTHON_TESTS})
  ADD_PYTHON_UNIT_TEST(${test} python/tests/${test} python)
ENDFOREACH()
