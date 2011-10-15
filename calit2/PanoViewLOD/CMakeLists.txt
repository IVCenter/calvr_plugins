SET(LIB_NAME PanoViewLOD)
SET(PLUGIN_HEADERS
    cube.hpp
    queue.hpp
    sph-cache.hpp
    sph-model.hpp
    tree.hpp
    glsl.h
    glyph.h
    math3d.h
    PanoDrawableLOD.h
    PanoViewLOD.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    cube.cpp
    sph-cache.cpp
    sph-model.cpp
    glsl.c
    glyph.c
    math3d.c
    PanoDrawableLOD.cpp
    PanoViewLOD.cpp
)

FIND_PACKAGE(GLEW)
FIND_PACKAGE(SDL)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${SDL_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${GLEW_LIBRARY})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${SDL_LIBRARY})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
