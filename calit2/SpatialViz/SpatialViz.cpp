#include "SpatialViz.h"

// CVR:
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>

// STD:
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <math.h>
#include <map>
#include <limits>
#include <iomanip>              // for formatting print statements

// OSG:
#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Vec3d>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osg/PrimitiveSet>
#include <osg/TexEnv>           // adding a texture state

// additional files
#include "Polyomino.hpp"        // to Generate Tetris2
#include "PuzzleGenerator.hpp"  // to Generate the puzzle pieces 


using namespace std;
using namespace cvr;
using namespace osg;

#ifdef HAVE_PHYSX
  using namespace physx;
#endif

CVRPLUGIN(SpatialViz)

// paths to puzzle models 
static string PATH_5X5 = "Puzzle5x5/Puzzle5x5.dae";
static string PATH_MAZE = "MazePuzzle/PuzzleMaze.dae";
static string PATH_MAZE_BOX = "MazePuzzle/MazeBox5.dae";
static string PATH_PUZZLE1 = "5PiecePuzzle/cube1of5.dae";
static string PATH_PUZZLE2 = "5PiecePuzzle/cube2of5.dae";
static string PATH_PUZZLE3 = "5PiecePuzzle/cube3of5.dae";
static string PATH_PUZZLE4 = "5PiecePuzzle/cube4of5.dae";
static string PATH_PUZZLE5 = "5PiecePuzzle/cube5of5.dae";



// PhysX 
#ifdef HAVE_PHYSX
PxPhysics *mPhysics = NULL;
PxScene *gScene = NULL;
PxScene *gScene2 = NULL;
PxScene *gSceneTetris = NULL;

PxScene *currScene = NULL;

PxReal myTimestep = 1.0f/60.0f;
PxReal currTime = 0.0f;
#endif

// -------------------------- Vectors of the OSG and PhysicX objects --------------------------

// used to cycle through the objects in the current scene -> will change based on the puzzle
vector<PositionAttitudeTransform*>* currSG;
vector<PxRigidDynamic*>* currPhysx;
    
// contain the starting positions of the objects to help reset the physics -> will change based on the puzzle
vector<osg::Vec3>* currStartingPositions;
vector<PxVec3>* currPhysxStartPos;   


// the objects for the Labyrinth
vector<PositionAttitudeTransform*> labyrinthObjs;
vector<PxRigidDynamic*> labyrinthPhysx;
vector<osg::Vec3> labyrinthStartingPos;
vector<physx::PxVec3> labyrinthPhysxStartPos;

// the objects for the 5x5 Puzzle
vector<PositionAttitudeTransform*> fiveObjs;
vector<PxRigidDynamic*> fivePhysx;
vector<osg::Vec3> fiveStartingPos;
vector<physx::PxVec3> fivePhysxStartPos;

// the objects for the Maze Puzzle
vector<PositionAttitudeTransform*> mazeObjs;
vector<PxRigidDynamic*> mazePhysx;
vector<osg::Vec3> mazeStartingPos;
vector<physx::PxVec3> mazePhysxStartPos;

// the objects for tetris
vector<PositionAttitudeTransform*> tetrisObjs;
vector<PxRigidDynamic*> tetrisPhysx;
vector<osg::Vec3> tetrisStartingPos;
vector<physx::PxVec3> tetrisPhysxStartPos;
Quat tetrisQuat;
int mainPieceMatchID;

// the objects for tetris2
vector<PositionAttitudeTransform*> tetrisObjs2;
vector<PxRigidDynamic*> tetrisPhysx2;
vector<osg::Vec3> tetrisStartingPos2;
vector<physx::PxVec3> tetrisPhysxStartPos2;
Quat tetrisQuat2;
int mainPieceMatchID2;

// to store the moving Scene Graph objects and their corresponding PhysX objects
vector<PositionAttitudeTransform*> movingObjs;
vector<PxRigidDynamic*> movingPhysx;


// alpha value for the cubes 
float cube_alpha = 1.0;
Vec4 cube_color = Vec4(1,1,1,1);


// ------------------------------------------ Start PhysX functions -------------------------------------


void SpatialViz::initPhysX()
{
    // --------------------------------------------- Initializing PhysX ----------------------------------------
#ifdef HAVE_PHYSX
    cerr << "--- initializing PhysX ---\n";
    static PxDefaultErrorCallback gDefaultErrorCallback;
    static PxDefaultAllocator gDefaultAllocatorCallback;
    static PxSimulationFilterShader gDefaultFilterShader = PxDefaultSimulationFilterShader;

    //cerr << "creating Foundation\n";
    PxFoundation *mFoundation = NULL;
    mFoundation = PxCreateFoundation( PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
 
    // create Physics object with the created foundation and with a 'default' scale tolerance.
    mPhysics = PxCreatePhysics( PX_PHYSICS_VERSION, *mFoundation, PxTolerancesScale());
    
    // -------------------- START checks --------------------
    if (!PxInitExtensions(*mPhysics)) cerr << "PxInitExtensions failed!" << endl;
    
    if(mPhysics == NULL) {
        cerr << "Error creating PhysX device." << endl << "Exiting..." << endl;
    }
    // -------------------- END checks --------------------
    
    // -------------------- Create the scene --------------------
    _sceneDesc = new PxSceneDesc(mPhysics->getTolerancesScale());
    _sceneDesc->gravity=PxVec3(0.0f, -9.81f, 0.0f); 
    
    if(!_sceneDesc->cpuDispatcher) 
    {
        PxDefaultCpuDispatcher* mCpuDispatcher = PxDefaultCpuDispatcherCreate(1);
        if(!mCpuDispatcher) {
            cerr << "PxDefaultCpuDispatcherCreate failed!" << endl;
        } 
        _sceneDesc->cpuDispatcher = mCpuDispatcher;
    }
    if(!_sceneDesc->filterShader)
        _sceneDesc->filterShader  = gDefaultFilterShader;
    
    // create three scenes
    gScene = mPhysics->createScene(*_sceneDesc);
    gScene2 = mPhysics->createScene(*_sceneDesc);
    gSceneTetris = mPhysics->createScene(*_sceneDesc);
    
    if (!gScene)
        cerr<<"createScene failed!"<<endl;
    
    // make scene for 5x5 and labyrinth
    gScene->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0);
    gScene->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
    
    // make scene for maze cube
    gScene2->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0);
    gScene2->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
    
    // make scene for tetris matching
    gSceneTetris->setVisualizationParameter(PxVisualizationParameter::eSCALE, 1.0);
    gSceneTetris->setVisualizationParameter(PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
    
    //cerr << "scenes created " << endl;
    
    
    // create a material: setting the coefficients of static friction, dynamic friction and restitution (how elastic a collision would be)
    PxMaterial* mMaterial = mPhysics->createMaterial(0.1,0.2,0.5);// also tried 0.0,0.0,0.5 -> same sticking problem with both
    
    // -------------------- Create ground plane ---------------------
    PxReal d = 0.0f;  
    PxTransform pose = PxTransform(PxVec3(0.0f, -0.25, 0.0f),PxQuat(PxHalfPi, PxVec3(0.0f, 0.0f, 1.0f)));
    PxRigidStatic* plane = mPhysics->createRigidStatic(pose);           // make the plane
    if (!plane)
        cerr << "create plane failed!" << endl;
    
    PxShape* shape = plane->createShape(PxPlaneGeometry(), *mMaterial); // adding material to the plane
    if (!shape) {
        cerr << "creating plane shape failed!" << endl;
    }
    gScene->addActor(*plane);                                           // add plane to the scene
    
    cerr << "--- PhysX Initialized ---" << endl;
#endif
    // ------------------------------------------ PhysX initialized -------------------------------------------
    
    cerr << "--- Genereating puzzles ---" << endl;
    // --------------------- Create the Labyrinth -------------------
    createLabyrinth(0.005, -0.245);                                     // boxHeight = 0.005 floorHeight = -0.245;
    cerr << "Created the Layrinth" << endl;
    
    // ------------------- Create the 5x5 Puzzle --------------------
    create5x5(5);
    cerr << "Created the 5x5" << endl;
    
    // ------------------- Create the Maze Puzzle -------------------
    createPuzzleCube(10);
    cerr << "Created the Puzzle Cube" << endl;
    
    // -------------------- Create Tetris Pieces --------------------
    createTetris(5);
    cerr << "Created the tetris pieces" << endl;
    
    // -------------------- Create Tetris2 Pieces -------------------
    createTetris2(5);
    cerr << "Created the second tetris" << endl;
    
    cerr << "--- Puzzles Generated ---" << endl;
    // ------------------------ Puzzles Made ------------------------
    
    // initialize the pointers
#ifdef HAVE_PHYSX
    currScene = gScene;
    currSG = &labyrinthObjs;
    currPhysx = &labyrinthPhysx;
    currStartingPositions = &labyrinthStartingPos;
    currPhysxStartPos = &labyrinthPhysxStartPos;
#else
    currSG = &labyrinthObjs;
    currStartingPositions = &labyrinthStartingPos;
#endif
    
}

//------------------------------------- Drawing the puzzles ------------------------------------------------------//

void SpatialViz::createTetris(int numPieces) {

    float dim = 0.025;                              // dimension of the cubes that make the tetris piece
    int size = floor(sqrt(numPieces)+1);            // dimension of the space the entire tetris piece will fit in
    cube_color = Vec4(0.15, 0.35, 0.75,1); cube_alpha = 1.0;    // color of first tetris piece
    
    // for the main piece
    vector<Vec3> mainTetris;
    mainPieceMatchID = (rand() % 4) - 1;
    cerr << "main piece MatchID = " << mainPieceMatchID << endl;
    
    // transform the piece
    PositionAttitudeTransform * fourTrans = new PositionAttitudeTransform();
    
    // set the rotation to 15 degrees (so can see better)
    Quat tetris4Quat = Quat(DegreesToRadians(15.0), Vec3(1,0,0));
    fourTrans->setAttitude(tetris4Quat);
    
    // create and draw 4 tetris pieces
    for (int puzzleNumber = -1; puzzleNumber < 3; puzzleNumber++) {
    
        // Generate the Tetris Piece
        vector<Vec3> cubeLocations;
        PuzzleGenerator::createTetrisPiece(size, numPieces, cubeLocations);
        
        if (mainPieceMatchID == puzzleNumber) {
            mainTetris = cubeLocations;
        }
        
        // draw the tetris piece
        currScene = gSceneTetris;
        for (int i = 0; i < cubeLocations.size(); i++) {  
            Vec3 pos = cubeLocations[i]; 
            pos *= dim*2;                               // scale the positions
            pos[0] += (puzzleNumber-0.5)*0.25;          // shift subsequent puzzles right
            pos[1] += 0.25;                             // shift puzzles up...
            createBoxes(1, PxVec3(pos[0],pos[1], pos[2]), PxVec3(dim, dim, dim), true, _objGroupTetris, &tetrisObjs, &tetrisPhysx, &tetrisStartingPos, &tetrisPhysxStartPos);
        }
        cube_color[0] += 0.15;                          // change the color for the next piece
    }
    
    _tetrisSwitch->addChild(fourTrans);
    fourTrans->addChild(_objGroupTetris);
    
    
    // draw the main cube
    cube_color = Vec4(0.15, 0.75, 0.35, 1); 
    
    // make a group for the piece
    _TetrisPiece = new Group;
    _mainTetrisSwitch = new Switch;
    
    // transform the piece
    PositionAttitudeTransform * pieceTrans = new PositionAttitudeTransform();
    
    // create a random axis
    float x,y,z;                        srand((unsigned)time(0));
    x = rand(); y = rand(); z = rand(); 
    PxVec3 axis = PxVec3(x,y,z);        axis.normalize();
    float angle = rand() % (330) + 15;         // angle ranges from 15 to 345
    
    // set the position and orientation 
    tetrisQuat = Quat(DegreesToRadians(angle), Vec3(axis[0], axis[2], axis[1]));
    pieceTrans->setAttitude(tetrisQuat);
    pieceTrans->setPosition(Vec3(0,0,-250));
    
    // draw the main tetris piece
    for (int i = 0; i < mainTetris.size(); i++) {  
        Vec3 pos = mainTetris[i]; 
        pos *= dim*2;                           // scale the positions
        createBoxes(1, PxVec3(pos[0],pos[1], pos[2]), PxVec3(dim, dim, dim), true, _TetrisPiece, &tetrisObjs, &tetrisPhysx, &tetrisStartingPos, &tetrisPhysxStartPos);
    }
    // add the group to the Tetris group
    _root->addChild(_mainTetrisSwitch);
    _mainTetrisSwitch->addChild(pieceTrans);
    pieceTrans->addChild(_TetrisPiece);
    
    // reset the color
    cube_color = Vec4(1,1,1,1); cube_alpha = 1.0;
} 


void SpatialViz::createTetris2(int numPieces) {

    _mainTetris2 = new Group();
    _mainTetrisSwitch2 = new Switch();
    
	// draw the main cube
	cube_color = Vec4(0.15, 0.75, 0.35, 1);
	float dim = 0.025;
	
	// generate the boxes for the tetris pieces
	int options = 4;
	vector<vector<vector<float> > > quiz = Polyomino::generatePolyominoQuiz(numPieces, options);
	for (int i = 0; i < quiz[0].size(); i++) {
	    vector<float> seg = quiz[0][i];
	    createBoxes(1, PxVec3(2 * dim * seg[0], 2 * dim * seg[1], 2 * dim * seg[2]), PxVec3(dim, dim, dim), true, _mainTetris2, &tetrisObjs2, &tetrisPhysx2, &tetrisStartingPos2, &tetrisPhysxStartPos2);
	}
    
    // draw the other 4 pieces
	cube_color = Vec4(0.15, 0.35, 0.75, 1); cube_alpha = 1.0;    // color of first tetris piece
	
	// randomize locations
	vector<int> pieceIndices = vector<int>();
	for (int i = 1; i < quiz.size(); i++) {
		pieceIndices.push_back(i);
	}
	random_shuffle(pieceIndices.begin(), pieceIndices.end());

	float spacing = 0.25;
	for (int i = 0; i < quiz.size() - 1; i++) {
		if (pieceIndices[i] == 1) mainPieceMatchID2 = i;
		for (int j = 0; j < quiz[pieceIndices[i]].size(); j++) {
	        vector<float> seg = quiz[pieceIndices[i]][j];
	        createBoxes(1, PxVec3(2 * dim * seg[0] + (i - 0.5) * spacing, 2 * dim * seg[1] + spacing, 2 * dim * seg[2]), PxVec3(dim, dim, dim), true, _TetrisPiece2, &tetrisObjs2, &tetrisPhysx2, &tetrisStartingPos2, &tetrisPhysxStartPos);
		}
		cube_color[0] += 0.15;
	}

	// randomize rotation of main piece
	PositionAttitudeTransform * pieceTrans = new PositionAttitudeTransform();
 	float x,y,z;                        
	srand((unsigned)time(0));
	x = rand(); y = rand(); z = rand();
 	PxVec3 axis = PxVec3(x,y,z);        axis.normalize();
	float angle = rand() % 360;
 	tetrisQuat2 = Quat(DegreesToRadians(angle), Vec3(axis[0], axis[2], axis[1]));
	pieceTrans->setAttitude(tetrisQuat2);
	pieceTrans->setPosition(Vec3(0,0,-250));
	
	// adding to root
    _root->addChild(_mainTetrisSwitch2);
    _mainTetrisSwitch2->addChild(pieceTrans);
    pieceTrans->addChild(_mainTetris2);
    
    // reset the color
	cube_color = Vec4(1, 1, 1, 1); cube_alpha = 1.0;
}


void SpatialViz::createPuzzleCube(int size) {
    
    // generate the positions for the Puzzle cube
	vector<Vec3> positions = vector<Vec3>();
	PuzzleGenerator::createPuzzleCube(size, positions);
	
	// draw the cubes
	currScene = gScene2;
	cube_color = Vec4(1.0, 0.2, 0.2, 1.0);
	for (int i = 0; i < positions.size(); i++) {
	    Vec3 pos = positions[i]*0.05;
	    createBoxes(1, PxVec3(pos[0], pos[1], pos[2]), PxVec3(0.025, 0.025, 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    }
    // draw the inner box
    cube_color = Vec4(1, 0.6, 0.2, 1.0);
    createBoxes(1, PxVec3(0.025 * (size - 1), 0.025 * (size - 1), 0.025 * (size - 1)), PxVec3(0.025 * (size - 2), 0.025 * (size - 2), 0.025 * (size - 2)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    
    // draw the outer boxes for collisions
    cube_color = Vec4(1,1,1,1);
    cube_alpha = 0.1;
    createBoxes(1, PxVec3(0.025 * (2*size), 0.025 * size - 0.025, 0.025 * size - 0.025), PxVec3(0.025, 0.025 * (size), 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    createBoxes(1, PxVec3(0.025 * (-2), 0.025 * size - 0.025, 0.025 * size - 0.025), PxVec3(0.025, 0.025 * (size), 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * (2*size), 0.025 * size - 0.025), PxVec3(0.025 * (size), 0.025, 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * (-2), 0.025 * size - 0.025), PxVec3(0.025 * (size), 0.025, 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * size - 0.025, 0.025 * (2*size)), PxVec3(0.025 * (size), 0.025 * (size), 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * size - 0.025, 0.025 * (-2)), PxVec3(0.025 * (size), 0.025 * (size), 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
    cube_alpha = 1.0;
    
    // place sphere in bottom left corner
    createSpheres(1, PxVec3(0, 0, 0), 0.023, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
}


void SpatialViz::create5x5(int puzzleSize)
{
    float puzzleDim = 0.25;
    float height = 0.05;    float radius = 0.0375;
    float width = 0.001;
    float spacing = 0.1;                        // to have the maze an even 5x5 grid
    PxVec3 position = PxVec3(0.0f, 0.0f, 0.0f);
    
    vector<Vec3> verticalWalls;
    vector<Vec3> horizontalWalls;
    vector<Vec3> floors;
    
    // generate the puzzle
    PuzzleGenerator::create5x5(puzzleSize, verticalWalls, horizontalWalls, floors);
    
    // draw the puzzle
    currScene = gScene;
    cube_alpha = 0.5;
    for (int level = 0; level <= puzzleSize; level++)
    {
        position.y = (level * spacing);
    
        // create the base and top
        if (level == 0 || level == puzzleSize) {
            for (int w = 0; w < puzzleSize; w++)
            {
                for (int l = 0; l < puzzleSize; l++) 
                {
                    // Base
                    if (level == 0)
                    createBoxes(1, PxVec3(puzzleDim - height - w*spacing, (level * spacing) - height, puzzleDim - height - l*spacing), PxVec3(height , width, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
                    // Top
                    if (level == puzzleSize && (w != 0 || l != 0)) // leave the top right open
                    createBoxes(1, PxVec3(puzzleDim - height - w*spacing, (level * spacing) - height, puzzleDim - height - l*spacing), PxVec3(height , width, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
                    
                }
            }
        }
        
        // maze section
        if (level < puzzleSize) {
            for (int w = 0; w <= puzzleSize; w++) 
            {
                for (int l = 0; l < puzzleSize; l++) 
                {   
                    // create the box for the outside walls
                    if ( w == 0 || w == puzzleSize ) {
                    
                        // vertical boxes
                        createBoxes(1, PxVec3(puzzleDim - w*spacing, (level * spacing), puzzleDim - height - l*spacing), PxVec3(width, height, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
                        
                        // horizontal boxes
                        if (level != 0 || w != puzzleSize || l != puzzleSize-1) {   // leave front left wall open
                            createBoxes(1, PxVec3(puzzleDim - height - l*spacing, (level * spacing), puzzleDim - w*spacing), PxVec3(height, height, width), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
                        }
                    }
                }
            }
        }
    }
    
    // add green grid for goal
    cube_color = Vec4(0.25, 0.75, 0.35, 0.0); cube_alpha = 1.0;
    createBoxes(1, PxVec3(puzzleDim - height - (puzzleSize-1)*spacing, (0 * spacing) - height, puzzleDim - height - puzzleSize*spacing), PxVec3(height , width, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
    cube_color = Vec4(1.0, 1.0, 1.0, 1.0); cube_alpha = 0.5;
    
    
    // draw the horizontal boxes
    for ( int i = 0; i < horizontalWalls.size(); i++ ) {
        Vec3 currBox = horizontalWalls[i];
        createBoxes(1, PxVec3(puzzleDim - height - currBox[0]*spacing, (currBox[2] * spacing), puzzleDim - currBox[1]*spacing), PxVec3(height, height, width), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
    }
    
    // draw the vertical boxes
    for ( int i = 0; i < verticalWalls.size(); i++) {
        Vec3 currBox = verticalWalls[i];
        createBoxes(1, PxVec3(puzzleDim - currBox[1]*spacing, (currBox[2] * spacing), puzzleDim - height - currBox[0]*spacing), PxVec3(width, height, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
    }
    
    /// draw the floors
    for ( int i = 0; i < floors.size(); i++) {
        Vec3 currBox = floors[i];
        createBoxes(1, PxVec3(puzzleDim - height - currBox[1]*spacing, (currBox[2] * spacing) - height, puzzleDim - height - currBox[0]*spacing), PxVec3(height , width, height), true, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
    }
    cube_alpha = 1.0;
    
    // place sphere on the top level 
    createSpheres(1, PxVec3(puzzleDim, (puzzleSize-1) * spacing, puzzleDim - height), radius, _objGroup, &fiveObjs, &fivePhysx, &fiveStartingPos, &fivePhysxStartPos);
        
}

void SpatialViz::createLabyrinth(float boxHeight, float floorHeight)
{      
    vector<PxVec3> positions;
    vector<PxVec3> dimensions;
    
    PuzzleGenerator::createLabyrinth(boxHeight, floorHeight, positions, dimensions);
    
    // draw the boxes for the labyrinth
    if (positions.size() != dimensions.size()) {
        cerr << "sizes don't match up!" << endl;
        cerr << "positions = " << positions.size() << " dim = " << dimensions.size() << endl;
    }
    currScene = gScene;
    for (int i = 0; i < positions.size(); i++) {
        createBoxes(1, positions.at(i), dimensions.at(i), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    }
    
    // draw the sphere for the labyrinth
    createSpheres(1, PxVec3(0.125, -0.245, 0.19), 0.0075, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
}

void SpatialViz::createBoxes(int num, PxVec3 startVec, PxVec3 dimensions, bool fixed, Group* parentGroup, vector<PositionAttitudeTransform*>* currSG, vector<PxRigidDynamic*>* currPhysX, vector<Vec3>* currStart, vector<PxVec3>* currStartPhysx, PxQuat quat) 
{
    // set the density, dimenstions, and material properties
    PxReal density = 1.0;
    PxBoxGeometry geometryBox(dimensions); 
    PxMaterial* mMaterial = mPhysics->createMaterial(0.1,0.2,0.5);;
    
    // create num cubes
    for (int i = 0; i < num; i++)
    {
        PxVec3 currVec = PxVec3(startVec.x, startVec.y+(i*0.12), startVec.z);
        PxTransform transform(currVec, quat);
        
        PxRigidDynamic *actor = PxCreateDynamic(*mPhysics, transform, geometryBox, *mMaterial, density);
        actor->setAngularDamping(0.75);
        actor->setLinearVelocity(PxVec3(0,0,0)); 
        
        // so cube doesn't "stick" when velocity = 0
        actor->setSleepThreshold(0.0);              
        
        if (fixed){
            actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
         }
        
        if (!actor)
            cerr << "create actor failed!" << endl;
        

        // add the box to the scene
        currScene->addActor(*actor);                                   
        currPhysX->push_back(actor); 
        
        // for restarting
        Vec3 start = Vec3(currVec.x, currVec.y, currVec.z);
        currStart->push_back(start);
        currStartPhysx->push_back(currVec);
        
        // add a cube to the given parent (root) then add the associated PAT to the moving Objs vector
        //                                              parent      center                          dimenstions                   transformation
        PositionAttitudeTransform * tempTrans = addCube(parentGroup, start, dimensions.x*2000, dimensions.z*2000, dimensions.y*2000, Vec3(0,0,0));   
        currSG->push_back(tempTrans);         

    }
}

void SpatialViz::createSpheres(int num, PxVec3 startVec, float radius, Group* parent, vector<PositionAttitudeTransform*>* currSG, vector<PxRigidDynamic*>* currPhysX, vector<Vec3>* currStart, vector<PxVec3>* currStartPhysx)
{
    // set the density, material and dimenstions
    PxReal density = 1.0f;
    PxMaterial* mMaterial = mPhysics->createMaterial(0.1,0.2,0.5);       // set the static, dynamic frictions and elasticity 
    PxSphereGeometry geometrySphere(radius);                             // make a sphere with the given radius
    
    for (int i = 0; i < num; i++)
    {
        //cerr << "Creating a sphere" << endl;
        
        PxVec3 currVec = PxVec3(startVec.x+(i*0.05), startVec.y, startVec.z);
        PxTransform transform(currVec, PxQuat::createIdentity());
        
        PxRigidDynamic *actor = PxCreateDynamic(*mPhysics, transform, geometrySphere, *mMaterial, density);
        actor->setAngularDamping(0.75);
        actor->setLinearVelocity(PxVec3(0,0,0)); 
        actor->setSleepThreshold(0.0);              // so sphere doesn't "stick" when velocity = 0
        
        if (!actor)
            cerr << "create sphere actor failed!" << endl;
        
        // add the sphere to the scene
        currScene->addActor(*actor);                                       
        currPhysX->push_back(actor); 
        
        // for restarting
        Vec3 start = Vec3(currVec.x, currVec.y, currVec.z);
        currStart->push_back(start);
        currStartPhysx->push_back(currVec);
        
        // adds a cube to the given parent (root) then add the associated PAT to the moving Objs vector
        PositionAttitudeTransform * tempTrans = addSphere(parent, Vec3(0,0,0), radius*1000, Vec3(0,0,0));   
        currSG->push_back(tempTrans); 
    }
}

// ------------------------------------------- End PhysX functions --------------------------------------

// ---------------------------------------- Start SpatialViz functions -----------------------------------
//constructor
SpatialViz::SpatialViz()
{

}

void SpatialViz::restartPhysics() 
{
    currTime = 0.0f;
    if (currSG == NULL) {
        cerr << "restartPhysics - no physics to restart" << endl;
        return;
    }
    if (*currSG == tetrisObjs || *currSG == tetrisObjs2){
        cerr << "restartPhysics - tetris object no restarting" << endl;
        return;
    }
    // reset the positions of the objects 
    // loop through the objects and update the positionAttitudeTransform based on the new location
    for(int i = 0; i < currSG->size(); i++)
    {
        //movingObjs[i]->setPosition(Vec3(0.0, 0.0, 0.0));
        currSG->at(i)->setPosition(currStartingPositions->at(i));
        
        //PxVec3 currVec = PxVec3(0.0, 0.3, 0.0);
        //PxTransform trans(currVec, PxQuat::createIdentity());
        
        PxTransform trans(currPhysxStartPos->at(i), PxQuat::createIdentity());
        currPhysx->at(i)->setGlobalPose(trans);
        
        currPhysx->at(i)->setLinearVelocity(PxVec3(0,0,0), true);
        currPhysx->at(i)->setAngularVelocity(PxVec3(0,0,0), true);
        
        //PxVec3 currVec = PxVec3(startVec.x+(i*0.05), startVec.y+(i*0.2), startVec.z);
        //PxTransform transform(currVec, PxQuat::createIdentity());
    
    }
}

void SpatialViz::menuCallback(MenuItem* menuItem)
{
    if (menuItem == _mazePuzzleButton)
    {
        cerr << "Maze Puzzle Button pressed" << endl;
        // shows the first puzzle only (maze)
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(0);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        
        currScene = gScene2;
        
        currSG = &mazeObjs;
        currPhysx = &mazePhysx;
        currStartingPositions = &mazeStartingPos;
        currPhysxStartPos = &mazePhysxStartPos;
        
        restartPhysics();
    }
    if (menuItem == _5x5puzzleButton)
    {
        cerr << "5x5 Puzzle Button pressed" << endl;  
        // shows the second puzzle only (5x5)
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(0);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1); 
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        
        currScene = gScene;
        
        currSG = &fiveObjs;
        currPhysx = &fivePhysx;
        currStartingPositions = &fiveStartingPos;
        currPhysxStartPos = &fivePhysxStartPos;
        
        restartPhysics();
    }
    if (menuItem == _labyrinthPuzzle)
    {
        cerr << "Labyrinth Puzzle Button pressed" << endl;
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(0); 
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        
        currScene = gScene;
        
        currSG = &labyrinthObjs;
        currPhysx = &labyrinthPhysx;
        currStartingPositions = &labyrinthStartingPos;
        currPhysxStartPos = &labyrinthPhysxStartPos;
        
        restartPhysics();
        
    }
    if (menuItem == _tetrisPuzzle)
    {
        cerr << "Tetris Puzzle Button pressed" << endl;
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(0);
        _mainTetrisSwitch->setSingleChildOn(0);
        
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        
        // currScene = gSceneTetris;
        currSG = &tetrisObjs;
        currPhysx = &tetrisPhysx;
        currStartingPositions = &tetrisStartingPos;
        currPhysxStartPos = &tetrisPhysxStartPos;
        
        restartPhysics();
    }
    if (menuItem == _tetrisPuzzle2)
    {
        cerr << "Cube Puzzle Button pressed" << endl;  
        // shows the second tetris puzzle
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        
        _tetrisSwitch2->setSingleChildOn(0);
        _mainTetrisSwitch2->setSingleChildOn(0);
        
        currSG = &tetrisObjs2;
        currPhysx = &tetrisPhysx2;
        currStartingPositions = &tetrisStartingPos2;
        currPhysxStartPos = &tetrisPhysxStartPos2;
        
    }  
    if (menuItem == _removePuzzles)
    {
        cerr << "Remove Puzzles " << endl;  
        // shows no puzzles
        
        resetSceneManager();
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        currSG = NULL;
    }
    if (menuItem == _restartPhysics)
    {
        cerr << "restart Physics" << endl;
        restartPhysics();
    }
  
}

// reset the Scene Manager so that rotations and scales are not transferred between puzzles
void SpatialViz::resetSceneManager(){

    Matrix m;
    SceneManager::instance()->setObjectMatrix(m);
    SceneManager::instance()->setObjectScale(1.0);
}

// intialize graphics and menus
bool SpatialViz::init()
{

	cerr << " -------------------- SpatialViz::SpatialViz -------------------- " << endl;

	// enable osg debugging
	//osg::setNotifyLevel( osg::INFO );
 	
	// --------------- create the menu ---------------
	_mainMenu = new SubMenu("SpatialViz", "SpatialViz");
 	_mainMenu->setCallback(this);
 	MenuSystem::instance()->addMenuItem(_mainMenu);

    _mazePuzzleButton = new MenuButton("Maze Puzzle");
    _mazePuzzleButton->setCallback(this);
    _mainMenu->addItem(_mazePuzzleButton);
    
	_5x5puzzleButton = new MenuButton("5x5 Puzzle");
    _5x5puzzleButton->setCallback(this);
    _mainMenu->addItem(_5x5puzzleButton);
    
    _labyrinthPuzzle = new MenuButton("Labyrinth Puzzle");
    _labyrinthPuzzle->setCallback(this);
    _mainMenu->addItem(_labyrinthPuzzle);
    
    _tetrisPuzzle = new MenuButton("Tetris Matching");
    _tetrisPuzzle->setCallback(this);
    _mainMenu->addItem(_tetrisPuzzle);
    
    _tetrisPuzzle2 = new MenuButton("Tetris 2 Puzzle");
    _tetrisPuzzle2->setCallback(this);
    _mainMenu->addItem(_tetrisPuzzle2);
        
    _removePuzzles = new MenuButton("Remove Puzzles");
    _removePuzzles->setCallback(this);
    _mainMenu->addItem(_removePuzzles);
        
	_restartPhysics = new MenuButton("Restart Physics");
    _restartPhysics->setCallback(this);
    _mainMenu->addItem(_restartPhysics);
	
	// --------------- create Nodes for the root and each of the puzzles ---------------
	_root = new Switch();
    _puzzleMazeGroup = new Group;
    _puzzle5x5Group = new Group;
    _piecePuzzleGroup = new Group;
    _labyrinthGroup = new Group;
    _labyrinthSwitch = new Switch;
    
    Vec3 scale = Vec3(25, 25, 25);
    
    // set up model path:
    //string configPath = ConfigManager::getEntry("Plugin.SpatialViz.DataDir");
    //cerr << configPath << endl;
    
	// --------------- add puzzles to _root ---------------
    _root->addChild(_puzzleMazeGroup);
    _root->addChild(_puzzle5x5Group);
    _root->addChild(_piecePuzzleGroup);
    _root->addChild(_labyrinthSwitch);
    _labyrinthSwitch->addChild(_labyrinthGroup);
     
    // 5x5 Puzzle
    _5x5Switch = new Switch;
	_objGroup = new Group;
	_root->addChild(_5x5Switch);
	_5x5Switch->addChild(_objGroup); 
	
	// Maze Puzzle
	_mazeSwitch = new Switch;
	_objGroupMaze = new Group;
	_root->addChild(_mazeSwitch);
	_mazeSwitch->addChild(_objGroupMaze);
	
	// Tetris Puzzle
	_tetrisSwitch = new Switch;
	_objGroupTetris = new Group;
	_root->addChild(_tetrisSwitch);
	
	// Second Tetris Puzzle
	_tetrisSwitch2 = new Switch;
	_TetrisPiece2 = new Group;
	_root->addChild(_tetrisSwitch2);
	_tetrisSwitch2->addChild(_TetrisPiece2);
	
	
	// ------------------------ Initialize Physics -------------------
	initPhysX();
	
 	// ----------------------- add each of the puzzles to the scene with scene objects --------------------------------
 	//                              name, navigation, movable, clip, context menu, show bounds
	soLab = new SceneObject("root", true, false, true, true, false);
	PluginHelper::registerSceneObject(soLab,"SpatialVizObject");
	//so->addChild(_root);
	soLab->addChild(_labyrinthSwitch);				// adding the _root to the scene
	soLab->attachToScene();				
	soLab->addMoveMenuItem();
	soLab->addNavigationMenuItem();
	//                              name, navigation, movable, clip, context menu, show bounds
	so5x5 = new SceneObject("5x5", true, false, true, true, false);
	PluginHelper::registerSceneObject(so5x5,"SpatialVizObject");
	so5x5->addChild(_5x5Switch);				// adding the _root to the scene
	so5x5->attachToScene();				
	so5x5->addMoveMenuItem();
	so5x5->addNavigationMenuItem();
	//                              name, navigation, movable, clip, context menu, show bounds
	soMaze = new SceneObject("Maze", true, false, true, true, false);
	PluginHelper::registerSceneObject(soMaze,"SpatialVizObject");
	soMaze->addChild(_mazeSwitch);				// adding the _root to the scene
	soMaze->attachToScene();				
	soMaze->addMoveMenuItem();
	soMaze->addNavigationMenuItem();
	//                                      name, navigation, movable, clip, context menu, show bounds
	soTetris = new SceneObject("tetris4", false, false, true, true, false);
	PluginHelper::registerSceneObject(soTetris,"SpatialVizObject");
	soTetris->addChild(_tetrisSwitch);				// adding the _root to the scene
	soTetris->attachToScene();				
	soTetris->addMoveMenuItem();
	soTetris->addNavigationMenuItem();
	//                                            name, navigation, movable, clip, context menu, show bounds
	soMainTetris = new SceneObject("mainTetris", true, true, true, true, false);
	PluginHelper::registerSceneObject(soMainTetris,"SpatialVizObject");
	soMainTetris->addChild(_mainTetrisSwitch);				// adding the _root to the scene
	soMainTetris->attachToScene();				
	soMainTetris->addMoveMenuItem();
	soMainTetris->addNavigationMenuItem();
    
    //                                      name, navigation, movable, clip, context menu, show bounds
	soTetris2 = new SceneObject("tetris4_2", false, false, true, true, false);
	PluginHelper::registerSceneObject(soTetris2,"SpatialVizObject");
	soTetris2->addChild(_tetrisSwitch2);				// adding the _root to the scene
	soTetris2->attachToScene();				
	soTetris2->addMoveMenuItem();
	soTetris2->addNavigationMenuItem();
	//                                            name, navigation, movable, clip, context menu, show bounds
	soMainTetris2 = new SceneObject("mainTetris_2", true, true, true, true, false);
	PluginHelper::registerSceneObject(soMainTetris2,"SpatialVizObject");
	soMainTetris2->addChild(_mainTetrisSwitch2);				// adding the _root to the scene
	soMainTetris2->attachToScene();				
	soMainTetris2->addMoveMenuItem();
	soMainTetris2->addNavigationMenuItem();
		
	
	// show no puzzles initially
    _5x5Switch->setSingleChildOn(1);
    _mazeSwitch->setSingleChildOn(1);
    _labyrinthSwitch->setSingleChildOn(1);  
    _tetrisSwitch->setSingleChildOn(1);
    _mainTetrisSwitch->setSingleChildOn(1);
    _tetrisSwitch2->setSingleChildOn(1);
    _mainTetrisSwitch2->setSingleChildOn(1);
    
    return true;
}


PositionAttitudeTransform * SpatialViz::addCube(Group * parent, Vec3 center, float dimX, float dimY, float dimZ, Vec3 trans)
{
    // add a cube to the root
	Box * unitCube = new Box(center, dimX, dimY, dimZ); // center, dimension
	ShapeDrawable * unitCubeDrawable = new ShapeDrawable(unitCube);
	
	// set the color
	unitCubeDrawable->setColor( cube_color );
	
	// move the cube 
	PositionAttitudeTransform * cubeTrans = new PositionAttitudeTransform();
	cubeTrans->setPosition(trans);
	
	_cubeGeode = new Geode();
	parent->addChild(cubeTrans);
	
	// set the Transparency 
	if (cube_alpha != 1.0)
	    setNodeTransparency((Node*)_cubeGeode, cube_alpha);
	
	cubeTrans->addChild(_cubeGeode);
	_cubeGeode->addDrawable(unitCubeDrawable);
	
	return cubeTrans;
}

PositionAttitudeTransform * SpatialViz::addSphere(Group * parent, Vec3 center, float radius, Vec3 trans)
{
    // unit sphere (radius = 25 to see)
	Sphere * unitSphere = new Sphere(center, radius);
	ShapeDrawable * unitSphereDrawable = new ShapeDrawable(unitSphere);
	
	// move the sphere 
	PositionAttitudeTransform * sphereTrans = new PositionAttitudeTransform();
	sphereTrans->setPosition(trans);
	
	_sphereGeode = new Geode();
	parent->addChild(sphereTrans);
	
	sphereTrans->addChild(_sphereGeode);
	_sphereGeode->addDrawable(unitSphereDrawable);
	
	return sphereTrans;
}

// set the transparency of the given node to alpha
void SpatialViz::setNodeTransparency(osg::Node *node, float alpha)
{
    osg::ref_ptr<osg::StateSet> stateset;
    stateset = node->getOrCreateStateSet();
    osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute(osg::StateAttribute::MATERIAL));
   
    if (!mm) mm = new osg::Material;
    mm->setAlpha(osg::Material::FRONT, alpha);

    stateset->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    stateset->setRenderingHint(alpha == 1.0 ?
                               osg::StateSet::OPAQUE_BIN :
                               osg::StateSet::TRANSPARENT_BIN);
    stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    node->setStateSet(stateset);
}


void SpatialViz::preFrame()
{
    currScene->simulate(myTimestep);   // advance the simulation by myTimestep
    currScene->fetchResults();

    // ---------- Update gravity vector: ----------
    osg::Matrix w2o = PluginHelper::getObjectMatrix();
    w2o.setTrans(Vec3(0,0,0));
    osg::Vec3 wDown, oDown, gravity;
    
    // Down in CalVr is (0,0,-1)
    wDown[0]=0;
    wDown[1]=0;
    wDown[2]=-1;
    oDown = w2o * wDown;
    oDown.normalize();
    oDown = oDown * 9.81;
    
    // rotate 90 about the x axis for PhysX
    gravity[0] = oDown[0];
    gravity[1] = oDown[2];
    gravity[2] = oDown[1];
    
    // update the gravity for the scene
    _sceneDesc->gravity=PxVec3(gravity[0], gravity[1], gravity[2]);
    currScene->setGravity(PxVec3(gravity[0], gravity[1], gravity[2]));  
    
    // if there are no SG objects -> return
    if (currSG == NULL){
        return;
    }
    
    // --------------------------- check for matches with Tetris ---------------------------
    if (*currSG == tetrisObjs) {
        bool orientationMatch = false;              bool positionMatch = false;
        // --------------------------- Matching Orientation --------------------------------
        Quat mainQuat = currSG->at(20)->getAttitude();
        mainQuat *= tetrisQuat;             // apply the arbitrary rotation applied to the main tetris piece
             
        // apply the world matrix rotation to the main tetris piece 
        Quat worldQuat;
        worldQuat.set(w2o);                 // world matrix as a quaternion
        mainQuat = worldQuat * mainQuat; 
        
        // get the angle and vector of the orientation of the tetris piece
        Vec3 mainVec; double mainAngle;
        mainQuat.getRotate(mainAngle, mainVec);
        
        // if the angle of the tetris piece is within 5 degrees of 0 -> match
        if (abs(mainAngle) < DegreesToRadians(5.0) || abs(mainAngle - DegreesToRadians(360.0)) < DegreesToRadians(5.0)){
            cerr << "---------------- ORENTATION MATCH ------------------ " << endl;
            orientationMatch = true;
        }
        else {
            cerr << endl;
            orientationMatch = false; 
        }
        
        // ---------------------------- Matching Position -----------------------------------
        // get the position of the main tetris piece
        Vec3 mainPos = soMainTetris->getPosition();         // tracks the position when translating (starts at 0,0,0)
        mainPos[2] -= 250;                                  // apply the same translations when it was created
        mainPos = worldQuat * mainPos;                      // apply the world matrix to the main tetris piece position
        
        // get the position of the matching tetris piece
        Vec3 tetrisPos = soTetris->getPosition();           // 0,0,0
        tetrisPos[0] += (mainPieceMatchID - 0.5) * 250;     // apply the same transformations to get the position of the matching piece
        tetrisPos[2] += 250;
        Matrix tetrisWorld = soTetris->getTransform();      
        tetrisPos = tetrisWorld * tetrisPos;                // apply the world matrix of the 4 tetris pieces to get the world position
        
        // if the x and z positions are within 25mm of each other -> match 
        if (abs(mainPos[0] - tetrisPos[0]) < 25.0 && abs(mainPos[2] - tetrisPos[2]) < 25.0) {
            cerr << "----------------- POSITION MATCH ------------------- " << endl;
            positionMatch = true;
        }
        else {
            cerr << endl;
            positionMatch = false;
        }
        if (orientationMatch && positionMatch) {
            cerr << "CORRECT MATCH" << endl;
        }
    }
    

    // --------------------------- check for matches with Tetris ---------------------------
    if (*currSG == tetrisObjs2) {
        bool orientationMatch = false;              bool positionMatch = false;
        // --------------------------- Matching Orientation --------------------------------
        Quat mainQuat = currSG->at(20)->getAttitude();
        mainQuat *= tetrisQuat2;             // apply the arbitrary rotation applied to the main tetris piece
             
        // apply the world matrix rotation to the main tetris piece 
        Quat worldQuat;
        worldQuat.set(w2o);                 // world matrix as a quaternion
        mainQuat = worldQuat * mainQuat; 
        
        // get the angle and vector of the orientation of the tetris piece
        Vec3 mainVec; double mainAngle;
        mainQuat.getRotate(mainAngle, mainVec);
        
        // if the angle of the tetris piece is within 5 degrees of 0 -> match
        if (abs(mainAngle) < DegreesToRadians(5.0) || abs(mainAngle - DegreesToRadians(360.0)) < DegreesToRadians(5.0)){
            cerr << "---------------- ORENTATION MATCH ------------------ " << endl;
            orientationMatch = true;
        }
        else {
            cerr << endl;
            orientationMatch = false; 
        }
        
        // ---------------------------- Matching Position -----------------------------------
        // get the position of the main tetris piece
        Vec3 mainPos = soMainTetris2->getPosition();         // tracks the position when translating (starts at 0,0,0)
        mainPos[2] -= 250;                                  // apply the same translations when it was created
        mainPos = worldQuat * mainPos;                      // apply the world matrix to the main tetris piece position
        
        // get the position of the matching tetris piece
        Vec3 tetrisPos = soTetris2->getPosition();           // 0,0,0
        tetrisPos[0] += (mainPieceMatchID2 - 0.5) * 250;     // apply the same transformations to get the position of the matching piece
        tetrisPos[2] += 250;
        Matrix tetrisWorld = soTetris2->getTransform();      
        tetrisPos = tetrisWorld * tetrisPos;                // apply the world matrix of the 4 tetris pieces to get the world position
        
        // if the x and z positions are within 25mm of each other -> match 
        if (abs(mainPos[0] - tetrisPos[0]) < 25.0 && abs(mainPos[2] - tetrisPos[2]) < 25.0) {
            cerr << "----------------- POSITION MATCH ------------------- " << endl;
            positionMatch = true;
        }
        else {
            cerr << endl;
            positionMatch = false;
        }
        if (orientationMatch && positionMatch) {
            cerr << "CORRECT MATCH" << endl;
        }
    }

    // --------------------------- update the physics if necessary ---------------------------
    if (currSG != NULL) {
        // loop through the objects and update the positionAttitudeTransform based on the new location
        for(int i = 0; i < currSG->size(); i++)
        {
            // the PhysX objects live in movingPhysx and the osg objects live in movingObjs
            PxTransform trans = currPhysx->at(i)->getGlobalPose();
            
            // osg down is z out/in is y PhysX is in meters and CalVR is in mm
            Vec3 nextPos = Vec3(1000*trans.p.x, 1000*trans.p.z, 1000*trans.p.y); 
            
            // update the osg position (and physX object for the forced fall)
            currSG->at(i)->setPosition(nextPos);
        }
        currTime += myTimestep;
    }
}

// this is called if the plugin is removed at runtime
SpatialViz::~SpatialViz()
{
   fprintf(stderr,"SpatialViz::~SpatialViz\n");
}

