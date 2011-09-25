SET(LIB_NAME LightEditor)
SET(PLUGIN_HEADERS
    LightEditor.h
    LightLoader.h
    LightManager.h
    LightShading.h
    Utility.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    LightEditor.cpp
    LightLoader.cpp
    LightManager.cpp
    LightShading.cpp
    Utility.cpp
)

FIND_PACKAGE(MXML)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MXML_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${MXML_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
