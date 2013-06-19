SET(LIB_NAME Maze2)
SET(PLUGIN_HEADERS
    Maze2.h
	MazeModelHandler.h
	AudioConfigHandler.h
	NavigationHandler.h
	MazeFileImporter.h
	MazeGenerator.h
	EOGClient/ECGClient.h
	EOGClient/CalibrationController.h
	EOGClient/BallHandler.h
	EOGClient/CaliFieldHandler.h
	EOGClient/Playback.h
        Audio/OSCPack.h
	AcoustiMaze/Pose.h
	AcoustiMaze/AcoustiMaze.h
	AcoustiMaze/CoordMatrix.h
	AcoustiMaze/MazePath.h
	AcoustiMaze/MazeSquare.h
)

ADD_LIBRARY(${LIB_NAME}
    "SHARED"
    ${PLUGIN_HEADERS}
    Maze2.cpp
	MazeModelHandler.cpp
	AudioConfigHandler.cpp
	NavigationHandler.cpp
	MazeFileImporter.cpp
	MazeGenerator.cpp
	MazeGeneratorExport.cpp
	EOGClient/ECGClient.cpp
	EOGClient/ECGClientUI.cpp
	EOGClient/CalibrationController.cpp
	EOGClient/BallHandler.cpp
	EOGClient/CaliFieldHandler.cpp
        Audio/OSCPack.cpp
	AcoustiMaze/Pose.cpp
	AcoustiMaze/AcoustiMaze.cpp
	AcoustiMaze/MazePath.cpp
	AcoustiMaze/MazeSquare.cpp
)

INCLUDE_DIRECTORIES(${OSG_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(${LIB_NAME} ${OSG_LIBRARIES})

ADD_CALVR_LIBRARIES(${LIB_NAME})

INSTALL(TARGETS ${LIB_NAME} DESTINATION lib/plugins)

