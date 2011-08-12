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


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
