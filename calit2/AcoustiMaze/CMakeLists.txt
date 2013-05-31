SET(LIB_NAME AcoustiMaze)
SET(PLUGIN_HEADERS
    AcoustiMaze.h
    CoordMatrix.h
    MazePath.h
    MazeSquare.h
    Pose.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    AcoustiMaze.cpp
    MazePath.cpp
    MazeSquare.cpp
    Pose.cpp
)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
