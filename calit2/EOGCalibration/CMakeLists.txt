SET(LIB_NAME EOGCalibration)
SET(PLUGIN_HEADERS
	EOGCalibration.h
	CalibrationController.h
	BallHandler.h
	CaliFieldHandler.h
	Playback.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
	EOGCalibration.cpp
	CalibrationController.cpp
	BallHandler.cpp
	CaliFieldHandler.cpp
)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)

