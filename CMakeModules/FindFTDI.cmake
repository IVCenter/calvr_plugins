FIND_PATH(FTDI_INCLUDE_DIR ftdi.h
  PATHS
  $ENV{FTDI_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(FTDI_INCLUDE_DIR ftdi.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(FTDI_LIBRARY 
  NAMES ftdi
  PATHS $ENV{FTDI_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(FTDI_LIBRARY 
  NAMES ftdi
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

SET(FTDI_FOUND "NO")
IF(FTDI_LIBRARY AND FTDI_INCLUDE_DIR)
  SET(FTDI_FOUND "YES")
ENDIF(FTDI_LIBRARY AND FTDI_INCLUDE_DIR)

