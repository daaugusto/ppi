cmake_minimum_required( VERSION 2.8 )

project( ppi )

#message("grammar file: ${GRAMMAR}, label: ${LABEL}")

if (NOT GRAMMAR)
   message(FATAL_ERROR "You need to specify the BNF grammar's path with -DGRAMMAR=<grammar path>")
endif ()

if (NOT EXISTS ${GRAMMAR})
   message(FATAL_ERROR "BNF grammar file '${GRAMMAR}' not found!")
endif ()

if (NOT LABEL)
   set(LABEL "main")
endif ()

if (NOT COST)
   set(COST "")
else()
   set(COST "const float COST[]${COST};") 
   message(STATUS "Using COST MATRIX (COST) as: ${COST}")
endif ()

if (NOT EF)
   if (COST)
      set(EF "COST[(int)X][(int)Y]")
   else ()
      set(EF "fabs((X)-(Y))")
   endif ()
endif ()
message(STATUS "Using ERROR FUNCTION (EF) as: ${EF}")

if (REDUCEMAX)
   message(STATUS "Reduce type: Maximum error")
endif ()

if (PROFILING)
   message(STATUS "Enabling time profiling")
   add_definitions(-DPROFILING)
endif ()

if (PROTECTED)
   message(STATUS "Enabling protected functions")
   add_definitions(-DPROTECTED)
endif ()

if (NATIVE)
   message(STATUS "Enabling OpenCL native functions")
   add_definitions(-DNATIVE)
endif ()

# In the following relative directory the assembled include files will be put in
set (INCLUDE_RELATIVE_DIR "${LABEL}-include")

set (GRAMMAR_INTERPRETER_DIR "${CMAKE_BINARY_DIR}/${INCLUDE_RELATIVE_DIR}")
file (MAKE_DIRECTORY "${GRAMMAR_INTERPRETER_DIR}")
#message("COMMAND python script/read_grammar.py -o ${CMAKE_BINARY_DIR} -g ${GRAMMAR} -i ${CMAKE_SOURCE_DIR}/problem/interpreter_core -p ${CMAKE_SOURCE_DIR}/problem/ppi_individual_print_function")
execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/script/read_grammar.py -o "${GRAMMAR_INTERPRETER_DIR}" -g ${GRAMMAR} -i ${CMAKE_SOURCE_DIR}/src/interpreter/interpreter_core -p ${CMAKE_SOURCE_DIR}/src/interpreter/ppi_individual_print_function
   OUTPUT_VARIABLE OUTPUT
   ERROR_VARIABLE ERROR
   RESULT_VARIABLE RESULT)

if (OUTPUT)
   message("\nOutput from [python3 ${CMAKE_SOURCE_DIR}/script/read_grammar.py -o ${GRAMMAR_INTERPRETER_DIR} -g ${GRAMMAR} -i ${CMAKE_SOURCE_DIR}/src/interpreter/interpreter_core -p ${CMAKE_SOURCE_DIR}/src/interpreter/ppi_individual_print_function]:\n ${OUTPUT}")
endif ()

if (NOT RESULT EQUAL 0)
   message(FATAL_ERROR "\nError, failed to run the script to build the grammar [python3 ${CMAKE_SOURCE_DIR}/script/read_grammar.py -o ${GRAMMAR_INTERPRETER_DIR} -g ${GRAMMAR} -i ${CMAKE_SOURCE_DIR}/src/interpreter/interpreter_core -p ${CMAKE_SOURCE_DIR}/src/interpreter/ppi_individual_print_function]:\n ${ERROR}")
endif ()

# Add the CMAKE_BINARY_DIR to the include path in order to make it easy to the
# source code to find the generated grammar and interpreter files
include_directories( "${GRAMMAR_INTERPRETER_DIR}" )
add_definitions( -DINCLUDE_RELATIVE_DIR=${INCLUDE_RELATIVE_DIR} )
add_definitions( -DLABEL=${LABEL} )

# The 'definitions.h' file holds macros and definitions which are taken from CMake's variables (such as the ERROR(X,Y) functional define)
configure_file( "${CMAKE_SOURCE_DIR}/src/definitions.h" "${CMAKE_BINARY_DIR}/${LABEL}-include/definitions.h")
configure_file( "${CMAKE_SOURCE_DIR}/src/costmatrix" "${CMAKE_BINARY_DIR}/${LABEL}-include/costmatrix")

add_subdirectory( src )
