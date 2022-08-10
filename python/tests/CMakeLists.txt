set(PYTHON_TESTS test_body.py test_dynamics.py test_goal.py
                 test_restrictions.py test_tools.py)

foreach(test ${PYTHON_TESTS})
  add_python_unit_test(${test} python/tests/${test} python)
endforeach()
