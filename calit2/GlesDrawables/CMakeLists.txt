SET(LIB_NAME GlesDrawables)

ADD_LIBRARY(${LIB_NAME}
    STATIC
    drawablesEntrance.cpp
    basicDrawables/basisRenderer.cpp
    basicDrawables/pointDrawable.cpp
    basicDrawables/strokeDrawable.cpp
    basicDrawables/planeDrawable.cpp)

add_definitions(-DNDEBUG)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(
    ${LIB_NAME}
    ${KERNEL_LIBRARIES}
)

ADD_CALVR_LIBRARIES(${LIB_NAME})


