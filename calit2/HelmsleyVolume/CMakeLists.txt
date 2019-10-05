SET(LIB_NAME HelmsleyVolume)

FIND_PACKAGE(DCMTK REQUIRED)

SET(EXT_LIBRARIES ${OSG_LIBRARIES} ${DCMTK_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(UNIX)
	FIND_PACKAGE(LibXml2 REQUIRED)
	SET(EXT_LIBRARIES ${EXT_LIBRARIES} ${LIBXML2_LIBRARIES})
	ADD_DEFINITIONS(-DHAVE_CONFIG_H)
endif(UNIX)

SET(PLUGIN_HEADERS
	VolumeMenu.h
    VolumeGroup.h
    HelmsleyVolume.h
    ImageLoader.hpp
	MeasurementTool.h
	ScreenshotTool.h
	Interactable.h
	FileSelector.h
)

if(WIN32)
	SET(PLUGIN_HEADERS
		${PLUGIN_HEADERS}
		dirent.h
	)
endif(WIN32)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
	VolumeMenu.cpp
    HelmsleyVolume.cpp
    ImageLoader.cpp
    VolumeGroup.cpp
	MeasurementTool.cpp
	ScreenshotTool.cpp
	FileSelector.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${DCMTK_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${EXT_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
