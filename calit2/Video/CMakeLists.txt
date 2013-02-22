SET(LIB_NAME Video)
SET(PLUGIN_HEADERS
    Video.h
    XMLConfig.h
    stringreplace.h
    texturemanager.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    Video.cpp
    XMLConfig.cxx
    stringreplace.cxx
    texturemanager.cxx
)

find_library(OPENAL_LIBRARY openal)
find_library(AVCODEC_LIBRARY avcodec)
find_library(AVDEVICE_LIBRARY avdevice)
find_library(SWSCALE_LIBRARY swscale)

FIND_PACKAGE(RSVP)
FIND_PACKAGE(GLEW)
include_directories(${RSVP_INCLUDE_DIR})
add_definitions( -D__STDC_CONSTANT_FORMATS -D__STDC_CONSTANT_MACROS)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME}
	${RSVP_LIBRARY}
	${GLEW_LIBRARY}
	${OPENAL_LIBRARY}
	${AVCODEC_LIBRARY}
	${AVDEVICE_LIBRARY}
	${SWSCALE_LIBRARY}
	${AVFORMAT_LIBRARY}
	${AVUTIL_LIBRARY}
)

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
