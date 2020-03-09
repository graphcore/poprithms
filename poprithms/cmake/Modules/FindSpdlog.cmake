set(WhatToDoString "Set SPDLOG_INSTALL_DIR during cmake configuration, \
by adding a flag like -DSPDLOG_INSTALL_DIR=/path-to-spdlog/ where \
path-to-spdlog contains spdlog/spdlog.h")

FIND_PATH(SPDLOG_INCLUDE_DIR
  NAMES spdlog/spdlog.h
  HINTS ${SPDLOG_INSTALL_DIR} 
        ${SPDLOG_INSTALL_DIR}/../include 
        ${SPDLOG_INSTALL_DIR}/include 
        ${CMAKE_INSTALL_PREFIX}/../spdlog/include
  PATH_SUFFIXES spdlog 
                spdlog/include
  DOC "directory with spdlog include files (spdlog/spdlog.h)")
IF(NOT SPDLOG_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "Could not set SPDLOG_INCLUDE_DIR. ${WhatToDoString}")
ENDIF()
MESSAGE(STATUS "Found SPDLOG_INCLUDE_DIR ${SPDLOG_INCLUDE_DIR}")
MARK_AS_ADVANCED(SPDLOG_INCLUDE_DIR)
