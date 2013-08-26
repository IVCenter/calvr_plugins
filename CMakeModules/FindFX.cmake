FIND_PATH(FX_INCLUDE_DIR FX.h
  PATHS
  $ENV{FX_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(FX_INCLUDE_DIR FX.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(FX_LIBRARY 
  NAMES FX
  PATHS $ENV{FX_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(FX_LIBRARY 
  NAMES FX
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

SET(FX_FOUND "NO")
IF(FX_LIBRARY AND FX_INCLUDE_DIR)
  SET(FX_FOUND "YES")
ENDIF(FX_LIBRARY AND FX_INCLUDE_DIR)

