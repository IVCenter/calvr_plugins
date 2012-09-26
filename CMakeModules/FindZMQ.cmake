FIND_PATH(ZMQ_INCLUDE_DIR zmq.h
  PATHS
  $ENV{ZMQ_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(ZMQ_INCLUDE_DIR zmq.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(ZMQ_LIBRARY 
  NAMES libzmq
  PATHS $ENV{ZMQ_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(ZMQ_LIBRARY 
  NAMES libzmq
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

SET(ZMQ_FOUND "NO")
IF(ZMQ_LIBRARY AND ZMQ_INCLUDE_DIR)
  SET(ZMQ_FOUND "YES")
ENDIF(ZMQ_LIBRARY AND ZMQ_INCLUDE_DIR)

