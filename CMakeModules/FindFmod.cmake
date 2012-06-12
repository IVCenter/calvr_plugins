FIND_PATH(FMOD_INCLUDE_DIR fmodex/fmod.h
  PATHS
  $ENV{FMOD_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include
)

FIND_PATH(FMOD_INCLUDE_DIR fmodex/fmod.h
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(FMOD_LIBRARY 
  NAMES fmodex64-4.40.00
  PATHS $ENV{FMOD_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)

FIND_LIBRARY(FMOD_LIBRARY 
  NAMES fmodex64-4.40.00
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

SET(FMOD_FOUND "NO")
IF(FMOD_LIBRARY AND FMOD_INCLUDE_DIR)
  SET(FMOD_FOUND "YES")
ENDIF(FMOD_LIBRARY AND FMOD_INCLUDE_DIR)

