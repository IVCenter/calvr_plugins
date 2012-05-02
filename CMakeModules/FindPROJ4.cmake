FIND_PATH(PROJ4_INCLUDE_DIR mxml.h
  PATHS
  $ENV{PROJ4_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(PROJ4_INCLUDE_DIR proj_api.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(PROJ4_LIBRARY 
  NAMES proj
  PATHS $ENV{PROJ4_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(PROJ4_LIBRARY 
  NAMES proj
  PATHS
    /usr/local
    /usr
    /sw
    /opt/local
    /opt/csw
    /opt
    /usr/freeware
  PATH_SUFFIXES lib64 lib
)

SET(PROJ4_FOUND "NO")
IF(PROJ4_LIBRARY AND PROJ4_INCLUDE_DIR)
  SET(PROJ4_FOUND "YES")
ENDIF(PROJ4_LIBRARY AND PROJ4_INCLUDE_DIR)

