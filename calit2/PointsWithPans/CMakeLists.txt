SET(LIB_NAME PointsWithPans)
SET(PLUGIN_HEADERS
    PointsWithPans.h
    PointsObject.h
    PanMarkerObject.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    PointsWithPans.cpp
    PointsObject.cpp
    PanMarkerObject.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
