SET(LIB_NAME TourCave)
SET(PLUGIN_HEADERS
    TourCave.h
    Audio/AssetManager.h
    Audio/oscpack.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    TourCave.cpp
    Audio/AssetManager.cpp
    Audio/oscpack.c
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
