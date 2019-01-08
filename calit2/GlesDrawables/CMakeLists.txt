SET(LIB_NAME GlesDrawables)

ADD_LIBRARY(${LIB_NAME}
    STATIC
    drawablesEntrance.cpp
    pano.cpp
    strokeDrawable.cpp
    planeDrawable.cpp
    quadDrawable.cpp
    pointDrawable.cpp)

add_definitions(-DNDEBUG)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(
    ${LIB_NAME}
    ${KERNEL_LIBRARIES}
)

ADD_CALVR_LIBRARIES(${LIB_NAME})


