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

target_include_directories(${LIB_NAME} PRIVATE
            ${ANDROID_NDK}/sources/third_party/vulkan/src/libs/glm)

TARGET_LINK_LIBRARIES(
    ${LIB_NAME}
    ${PHYSX_LIBS}
    ${KERNEL_LIBRARIES}
)

ADD_CALVR_LIBRARIES(${LIB_NAME})


