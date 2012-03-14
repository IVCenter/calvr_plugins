SET(LIB_NAME Sketch)
SET(PLUGIN_HEADERS
    Sketch.h
    SketchObject.h
    SketchLine.h
    SketchRibbon.h
    ColorSelector.h
    SketchShape.h
    Layout.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    Sketch.cpp
    SketchLine.cpp
    SketchRibbon.cpp
    ColorSelector.cpp
    SketchShape.cpp
    Layout.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
