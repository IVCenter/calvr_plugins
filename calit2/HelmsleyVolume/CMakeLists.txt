SET(LIB_NAME HelmsleyVolume)

FIND_PACKAGE(DCMTK REQUIRED COMPONENTS ofstd oflog dcmdata dcmimgle)
FIND_PACKAGE(OPEN_VR)
#FIND_PACKAGE(YAML-CPP REQUIRED)
 

SET(PLUGIN_HEADERS
	VolumeMenu.h
    VolumeGroup.h
    HelmsleyVolume.h
    ImageLoader.hpp
	MeasurementTool.h
	ScreenshotTool.h
	Interactable.h
	FileSelector.h
	UIExtensions.h
	CuttingPlane.h
	Utils.h
	CurvedMenu.h
	CenterlineTool.h
)

if(WIN32)
	SET(PLUGIN_HEADERS
		${PLUGIN_HEADERS}
		dirent.h
	)
endif(WIN32)

IF(OPENVR_FOUND)
    MESSAGE( STATUS "OpenVR found" )
    ADD_DEFINITIONS(-DWITH_OPENVR)
ENDIF(OPENVR_FOUND)

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
	UIExtensions.cpp
	CuttingPlane.cpp
	Utils.cpp
	CurvedMenu.cpp
	CenterlineTool.cpp
)


INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})

SET(EXT_LIBRARIES ${OSG_LIBRARIES} ${DCMTK_LIBRARIES})
INCLUDE_DIRECTORIES(${DCMTK_INCLUDE_DIRS})


if(UNIX)
	FIND_PACKAGE(LibXml2 REQUIRED)
	SET(EXT_LIBRARIES ${EXT_LIBRARIES} ${LIBXML2_LIBRARIES})
	ADD_DEFINITIONS(-DHAVE_CONFIG_H)
endif(UNIX)

IF(OPENVR_FOUND)
    INCLUDE_DIRECTORIES(${OPENVR_INCLUDE_DIRS})
    SET(EXT_LIBRARIES ${EXT_LIBRARIES} ${OPENVR_LIBRARY})
ENDIF(OPENVR_FOUND)

#TARGET_LINK_LIBRARIES(${LIB_NAME} ${EXT_LIBRARIES} yaml-cpp opengl32)

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)
