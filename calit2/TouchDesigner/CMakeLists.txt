SET(LIB_NAME TouchDesigner)
SET(PLUGIN_HEADERS
    TouchDesigner.h
    shapes/BasicShape.h
    shapes/CircleShape.h
    shapes/TriangleShape.h
    shapes/RectShape.h
    shapes/BoxShape.h
    shapes/PyramidShape.h
    shapes/SphereShape.h
    util/vvtokenizer.h
    util/ShapeHelper.h
    socket/SocketThread.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    TouchDesigner.cpp
    shapes/BasicShape.cpp
    shapes/CircleShape.cpp
    shapes/TriangleShape.cpp
    shapes/BoxShape.cpp
    shapes/PyramidShape.cpp
    shapes/SphereShape.cpp
    shapes/RectShape.cpp
    util/vvtokenizer.cpp
    util/ShapeHelper.cpp
    socket/SocketThread.cpp    
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
