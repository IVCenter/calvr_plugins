FIND_PATH(SNAPPY_INCLUDE_DIR snappy.h
  PATHS
  $ENV{SNAPPY_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(SNAPPY_INCLUDE_DIR snappy.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(SNAPPY_LIBRARY 
  NAMES libsnappy
  PATHS $ENV{SNAPPY_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(SNAPPY_LIBRARY 
  NAMES libsnappy
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

SET(SNAPPY_FOUND "NO")
IF(SNAPPY_LIBRARY AND SNAPPY_INCLUDE_DIR)
  SET(SNAPPY_FOUND "YES")
ENDIF(SNAPPY_LIBRARY AND SNAPPY_INCLUDE_DIR)

