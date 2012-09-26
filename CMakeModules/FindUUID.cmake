FIND_PATH(UUID_INCLUDE_DIR uuid.h
  PATHS
  $ENV{UUID_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(UUID_INCLUDE_DIR uuid.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(UUID_LIBRARY 
  NAMES libuuid
  PATHS $ENV{UUID_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(UUID_LIBRARY 
  NAMES libuuid
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

SET(UUID_FOUND "NO")
IF(UUID_LIBRARY AND UUID_INCLUDE_DIR)
  SET(UUID_FOUND "YES")
ENDIF(UUID_LIBRARY AND UUID_INCLUDE_DIR)

