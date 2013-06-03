FIND_PATH(Collider_pp_DIR ColliderPlusPlus.hpp
  PATHS
  $ENV{COLLIDER_HOME}
  NO_DEFAULT_PATH
    PATH_SUFFIXES include/ColliderPlusPlus
)

FIND_PATH(Collider_pp_DIR ColliderPlusPlus.hpp
  PATHS
  /usr/local/include
  /usr/include
  /sw/include # Fink
  /opt/local/include # DarwinPorts
  /opt/csw/include # Blastwave
  /opt/include
)

FIND_LIBRARY(Collider_pp_LIBRARY 
  NAMES collider++
  PATHS $ENV{COLLIDER_HOME}
    NO_DEFAULT_PATH
    PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(Collider_pp_LIBRARY 
  NAMES collider++
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

SET(COLLIDER_FOUND "NO")
IF(Collider_pp_LIBRARY AND Collider_pp_DIR)
  SET(COLLIDER_FOUND "YES")
ENDIF(Collider_pp_LIBRARY AND Collider_pp_DIR)

