SET(LIB_NAME EZVis)

SET(PLUGIN_HEADERS
    EZVis.h
    Globals.h
    SocketThread.h
    ThreadQueue.h
    ThreadMap.h
    CommandParser.h
    MainNode.h
    zhelpers.h
    Router.h
    Type.h
    Vec3Type.h
    Vec4Type.h
    StringType.h
    FloatType.h
    BoolType.h
    shapes/GeometryShape.h
    shapes/BasicShape.h
    shapes/Factory.h
    shapes/PointShape.h
    shapes/QuadShape.h
    shapes/ShapeFactory.h
    shapes/TriangleShape.h
    shapes/CircleShape.h
    shapes/LineShape.h
    shapes/ScalableLineShape.h
    shapes/RectangleShape.h
    shapes/TextShape.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    EZVis.cpp
    SocketThread.cpp
    ThreadQueue.cpp
    ThreadMap.cpp
    CommandParser.cpp
    MainNode.cpp
    Router.cpp
    Vec3Type.cpp
    Vec4Type.cpp
    StringType.cpp
    FloatType.cpp
    BoolType.cpp
    shapes/GeometryShape.cpp
    shapes/BasicShape.cpp
    shapes/PointShape.cpp
    shapes/QuadShape.cpp
    shapes/TriangleShape.cpp
    shapes/CircleShape.cpp
    shapes/LineShape.cpp
    shapes/ScalableLineShape.cpp
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
