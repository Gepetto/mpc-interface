SET(PYTHON_SOURCES
  __init__.py
  body.py
  combinations.py
  dynamics.py
  goal.py
  restrictions.py
  tools.py
  )

FOREACH(source ${PYTHON_SOURCES})
  PYTHON_INSTALL_ON_SITE(mpc_interface ${source})
ENDFOREACH(source)

IF(BUILD_TESTING)
  ADD_SUBDIRECTORY(tests)
ENDIF()
