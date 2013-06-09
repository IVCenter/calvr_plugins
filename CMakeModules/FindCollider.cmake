FIND_PATH(COLLIDER_DIR Collider.hpp
PATHS
$ENV{COLLIDER_HOME}
NO_DEFAULT_PATH
PATH_SUFFIXES include/collider
)

FIND_PATH(COLLIDER_DIR Collider.hpp
PATHS
/usr/local/include
/usr/include
/sw/include # Fink
/opt/local/include # DarwinPorts
/opt/csw/include # Blastwave
/opt/include
PATH_SUFFIXES collider
)

FIND_LIBRARY(COLLIDER_LIBRARY 
NAMES collider
PATHS $ENV{COLLIDER_HOME}
NO_DEFAULT_PATH
PATH_SUFFIXES lib64 lib
)
FIND_LIBRARY(COLLIDER_LIBRARY 
NAMES collider
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
IF(COLLIDER_LIBRARY AND COLLIDER_DIR)
  SET(COLLIDER_FOUND "YES")
ENDIF(COLLIDER_LIBRARY AND COLLIDER_DIR)

