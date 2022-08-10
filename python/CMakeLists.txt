set(PYTHON_SOURCES
    __init__.py
    body.py
    combinations.py
    dynamics.py
    goal.py
    restrictions.py
    tools.py)

foreach(source ${PYTHON_SOURCES})
  python_install_on_site(mpc_interface ${source})
endforeach(source)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
