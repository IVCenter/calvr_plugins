FIND_PATH(FTD2XX_INCLUDE_DIR libftd2xx.h
  PATHS
  $ENV{FTD2XX_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(FTD2XX_INCLUDE_DIR ftd2xx.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(FTD2XX_LIBRARY 
  NAMES ftd2xx 
  PATHS $ENV{FTD2XX_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(FTD2XX_LIBRARY 
  NAMES ftd2xx
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

SET(FTD2XX_FOUND "NO")
IF(FTD2XX_LIBRARY AND FTD2XX_INCLUDE_DIR)
  SET(FTD2XX_FOUND "YES")
ENDIF(FTD2XX_LIBRARY AND FTD2XX_INCLUDE_DIR)

