SET(LIB_NAME GreenLight)
SET(PLUGIN_HEADERS
    GreenLight.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    GreenLight.cpp
    LoadEntities.cpp
    Intersection.cpp
    Entity.cpp
    Hardware.cpp
    Power.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
