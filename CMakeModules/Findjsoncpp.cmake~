FIND_PATH(JSONCPP_INCLUDE_DIR jsoncpp/json/json.h
  PATHS
  $ENV{JSONCPP_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(JSONCPP_INCLUDE_DIR jsoncpp/json/json.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(JSONCPP_LIBRARY 
  NAMES jsoncpp
  PATHS $ENV{JSONCPP_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(JSONCPP_LIBRARY 
  NAMES jsoncpp
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

SET(JSONCPP_FOUND "NO")
IF(JSONCPP_LIBRARY AND JSONCPP_INCLUDE_DIR)
  SET(JSONCPP_FOUND "YES")
ENDIF(JSONCPP_LIBRARY AND JSONCPP_INCLUDE_DIR)

