SET(LIB_NAME PhysxBall)
SET(PLUGIN_HEADERS
    PhysxBall.h
    #pointDrawable.h
)
SET(PHYX_HEADERS
    PhysicsUtils.h
    Engine.h)
SET(PHYX_CPP
    PhysicsUtils.cpp
    Engine.cpp)

ADD_LIBRARY(${LIB_NAME}
    STATIC
    PhysxBall.cpp
    #pointDrawable.cpp
    ${PLUGIN_HEADERS}
    ${PHYX_CPP}
)

add_definitions(-DNDEBUG)
add_definitions(-DHAVE_PHYSX)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PHYSX_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(
    ${LIB_NAME}
    ${PHYSX_LIBS}
    ${KERNEL_LIBRARIES}
)

#set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER "PhysicsUtils.h;Engine.h;PhysxBall.h")
IF(ANDROID)
	file(GLOB PUBLIC_HEADERS
		"./*.h"
	)
	file(COPY ${PUBLIC_HEADERS} DESTINATION "../../include")
ENDIF(ANDROID)
ADD_CALVR_LIBRARIES(${LIB_NAME})


