FIND_PATH(PROTOBUF_INCLUDE_DIR config.h
  PATHS
  $ENV{PROTOBUF_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(PROTOBUF_INCLUDE_DIR config.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(PROTOBUF_LIBRARY 
  NAMES libprotobuf
  PATHS $ENV{PROTOBUF_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(PROTOBUF_LIBRARY 
  NAMES libprotobuf
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

SET(PROTOBUF_FOUND "NO")
IF(PROTOBUF_LIBRARY AND PROTOBUF_INCLUDE_DIR)
  SET(PROTOBUF_FOUND "YES")
ENDIF(PROTOBUF_LIBRARY AND PROTOBUF_INCLUDE_DIR)

