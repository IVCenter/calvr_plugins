SET(LIB_NAME Helmsley)

SET(GLM_DIRECTORY "D:/OpenGL-Libs/glm")
include_directories(${GLM_DIRECTORY})

ADD_LIBRARY(${LIB_NAME}
    STATIC
    VolumeViewer.cpp
    dcmRenderer.cpp
    cubeDrawable.cpp)



add_definitions(-DNDEBUG)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

TARGET_LINK_LIBRARIES(
    ${LIB_NAME}
    ${KERNEL_LIBRARIES}
)

ADD_CALVR_LIBRARIES(${LIB_NAME})


