SET(LIB_NAME Mugic)

SET(PLUGIN_HEADERS
    Mugic.h
    SocketThread.h
    ThreadQueue.h
    ThreadMap.h
    #Variables.h
    CommandParser.h
    MainNode.h
    zhelpers.h
    shapes/BasicShape.h
    shapes/Factory.h
    shapes/PointShape.h
    shapes/QuadShape.h
    shapes/ShapeFactory.h
    shapes/TriangleShape.h
    shapes/CircleShape.h
    shapes/LineShape.h
    shapes/RectangleShape.h
    shapes/TextShape.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    Mugic.cpp
    SocketThread.cpp
    ThreadQueue.cpp
    ThreadMap.cpp
    #Variables.cpp
    CommandParser.cpp
    MainNode.cpp
    shapes/BasicShape.cpp
    shapes/PointShape.cpp
    shapes/QuadShape.cpp
    shapes/TriangleShape.cpp
    shapes/CircleShape.cpp
    shapes/LineShape.cpp
    shapes/RectangleShape.cpp
    shapes/TextShape.cpp
)

FIND_PACKAGE(ZMQ REQUIRED)

INCLUDE_DIRECTORIES( ${ZMQ_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${ZMQ_LIBRARY})

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)

ADD_SUBDIRECTORY(router)
