#include "SpatialViz.h"

#include <cvrKernel/PluginHelper.h>

#include <PluginMessageType.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <math.h>

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


#include "Polyomino.hpp"        // to Generate Tetris2


#include <osg/TexEnv>           // adding a texture state

#include <map>
#include <limits>
#include <iomanip>              // for formatting print statements

using namespace std;
using namespace cvr;
using namespace osg;
using namespace physx;

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
PxPhysics *mPhysics = NULL;
PxScene *gScene = NULL;
PxScene *gScene2 = NULL;
PxScene *gSceneTetris = NULL;

PxScene *currScene = NULL;

PxReal myTimestep = 1.0f/60.0f;
PxReal currTime = 0.0f;
PxReal end = 1.0f;

// used to cycle through the objects in the current scene -> will change based on the puzzle
vector<PositionAttitudeTransform*>* currSG;
vector<PxRigidDynamic*>* currPhysx;
    
// contain the starting positions of the objects to help reset the physics -> will change based on the puzzle
std::vector<osg::Vec3>* currStartingPositions;
std::vector<physx::PxVec3>* currPhysxStartPos;   


// the objects for the Labyrinth
vector<PositionAttitudeTransform*> labyrinthObjs;
vector<PxRigidDynamic*> labyrinthPhysx;

std::vector<osg::Vec3> labyrinthStartingPos;
std::vector<physx::PxVec3> labyrinthPhysxStartPos;

// the objects for the 5x5 Puzzle
vector<PositionAttitudeTransform*> fiveObjs;
vector<PxRigidDynamic*> fivePhysx;

std::vector<osg::Vec3> fiveStartingPos;
std::vector<physx::PxVec3> fivePhysxStartPos;

// the objects for the Maze Puzzle
vector<PositionAttitudeTransform*> mazeObjs;
vector<PxRigidDynamic*> mazePhysx;

std::vector<osg::Vec3> mazeStartingPos;
std::vector<physx::PxVec3> mazePhysxStartPos;

// the objects for the tetris
vector<PositionAttitudeTransform*> tetrisObjs;
vector<PxRigidDynamic*> tetrisPhysx;

std::vector<osg::Vec3> tetrisStartingPos;
std::vector<physx::PxVec3> tetrisPhysxStartPos;


// the objects for the tetris
vector<PositionAttitudeTransform*> tetrisObjs2;
vector<PxRigidDynamic*> tetrisPhysx2;

std::vector<osg::Vec3> tetrisStartingPos2;
std::vector<physx::PxVec3> tetrisPhysxStartPos2;


// to store the moving Scene Graph objects and their corresponding PhysX objects
vector<PositionAttitudeTransform*> movingObjs;
vector<PxRigidDynamic*> movingPhysx;



// alpha value for the cubes 
float cube_alpha = 1.0;
Vec4 cube_color = Vec4(1,1,1,1);



// Code for character controller
static PxControllerManager* manager = NULL;
PxController* characterController = NULL;
bool xTrue = false, yTrue = false, zTrue = false;
PxReal x = 0, z = 0, y = 0;
PxReal movement = 0.2f;
static PxF32 startElapsedTime;

// ------------------------------------------ Start PhysX functions -------------------------------------

void SpatialViz::initPhysX()
{
    // ------------------------ PhysX -------------------
    cerr << "--- initializing PhysX ---\n";
    static PxDefaultErrorCallback gDefaultErrorCallback;
    static PxDefaultAllocator gDefaultAllocatorCallback;
    static PxSimulationFilterShader gDefaultFilterShader = PxDefaultSimulationFilterShader;

    //PxRigidBody *box;       // was PxRigidActor
 
    cerr << "creating Foundation\n";
    PxFoundation *mFoundation = NULL;
    mFoundation = PxCreateFoundation( PX_PHYSICS_VERSION, gDefaultAllocatorCallback, gDefaultErrorCallback);
 
    cerr <<"creating Physics\n";
    // create Physics object with the created foundation and with a 'default' scale tolerance.
    mPhysics = PxCreatePhysics( PX_PHYSICS_VERSION, *mFoundation, PxTolerancesScale());
    
    // extension check
    if (!PxInitExtensions(*mPhysics)) cerr << "PxInitExtensions failed!" << endl;

    // testing
    if(mPhysics == NULL) 
    {
        cerr << "Error creating PhysX device." << endl;
        cerr << "Exiting..." << endl;
    }
   
    // -------------------- Create the scene --------------------
    cerr << "creating the scene\n";
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
    
    // create the three scenes
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
    
    cerr << "scenes created " << endl;
    
    
    // create a material: setting the coefficients of static friction, dynamic friction and 
    // restitution (how elastic a collision would be)
    PxMaterial* mMaterial = mPhysics->createMaterial(0.1,0.2,0.5);// also tried 0.0,0.0,0.5 -> same sticking problem with both
    
    // -------------------- Create ground plane --------------------
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
    
    cerr << "Created ground plane" << endl;
    
    
    // --------------------- Create the Labyrinth ------------------
    createLabyrinth(0.005, -0.245);                                     // boxHeight = 0.005 floorHeight = -0.245;
    cerr << "Created the Layrinth" << endl;
    
    // ------------------- Create the 5x5 Puzzle -------------------
    create5x5(5);
    cerr << "Created the 5x5" << endl;
    
    // ------------------- Create the Maze Puzzle ------------------
    createPuzzleCube(10);
    cerr << "Created the Puzzle Cube" << endl;
    
    // ------------------- Create the Tetris Pieces ---------------
    createTetris(5);
    cerr << "Created the tetris pieces" << endl;
    
    // tetris 2
    createTetris2(5);
    cerr << "Created the second tetris" << endl;
    
    currScene = gScene;
    currSG = &labyrinthObjs;
    currPhysx = &labyrinthPhysx;
    currStartingPositions = &labyrinthStartingPos;
    currPhysxStartPos = &labyrinthPhysxStartPos;
   
    
    cerr << "--- DONE initializing PhysX ---\n";
} 

void SpatialViz::createTetris(int numPieces) {

    float dim = 0.025;                          // dimension of the cubes that make the tetris piece
    int size = floor(sqrt(numPieces)+1);          // dimension of the space the entire tetirs piece will fit in
    cube_color = Vec4(0.15, 0.35, 0.75,1); cube_alpha = 1.0;    // color of first tetris piece
    
    // for the main piece
    vector<Vec3> mainTetris;
    int mainPiece = (rand() % 4) - 1;
    
    // transform the piece
    PositionAttitudeTransform * fourTrans = new PositionAttitudeTransform();
    // set the rotation
    Quat tetris4Quat = Quat(DegreesToRadians(15.0), Vec3(1,0,0));
    //tetris4Quat*= Quat(DegreesToRadians(15.0), Vec3(0,0,1));
    fourTrans->setAttitude(tetris4Quat);
    //fourTrans->setPosition(Vec3(0,0,-250));
    
    // create 4 tetris pieces
    for (int puzzleNumber = -1; puzzleNumber < 3; puzzleNumber++) {
        // make the tetris piece
        vector<Vec3> cubeLocations;
        createTetrisPiece(size, numPieces, cubeLocations);
        
        if (mainPiece == puzzleNumber) {
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
    
    // create a random axis
    float x,y,z;
    srand((unsigned)time(0));
    x = rand(); y = rand(); z = rand();
    PxVec3 axis = PxVec3(x,y,z);
    axis.normalize();
    float angle = rand() % 360;
    cerr << "axis: " << axis[0] << " : " << axis[1] << " : " << axis[2] << endl;
    
    // make a group for the piece
    _TetrisPiece = new Group;
    _mainTetrisSwitch = new Switch;
        
    // transform the piece
    PositionAttitudeTransform * pieceTrans = new PositionAttitudeTransform();
    // set the rotation
    Quat tetrisQuat = Quat(DegreesToRadians(angle), Vec3(axis[0], axis[2], axis[1]));
    pieceTrans->setAttitude(tetrisQuat);
    pieceTrans->setPosition(Vec3(0,0,-250));
        
    for (int i = 0; i < mainTetris.size(); i++) {  
        Vec3 pos = mainTetris[i]; 
        pos *= dim*2;                           // scale the positions
        //pos[1] -= 1.5*0.25;                     // shift main piece down
        
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
	vector<vector<vector<float> > > quiz = Polyomino::generatePolyominoQuiz(numPieces, 4);
	for (int i = 0; i < quiz[0].size(); i++) {
	    vector<float> seg = quiz[0][i];
	    createBoxes(1, PxVec3(2 * dim * seg[0], 2 * dim * seg[1], 2 * dim * seg[2]), PxVec3(dim, dim, dim), true, _mainTetris2, &tetrisObjs2, &tetrisPhysx2, &tetrisStartingPos2, &tetrisPhysxStartPos2);
	    cerr << "check : " << 2 * dim * seg[0]<< " : "<< 2 * dim * seg[1] << " : " << 2 * dim * seg[2]<< endl;
	}
    cerr << "made main" << endl;
	// draw the other pieces
	cube_color = Vec4(0.15, 0.35, 0.75, 1); cube_alpha = 1.0;    // color of first tetris piece
	float spacing = 2*dim * ((int)sqrt(numPieces) + 2);
	cerr << "spacing = " << spacing << endl;
	for (int i = 1; i < quiz.size(); i++) {
		for (int j = 0; j < quiz[i].size(); j++) {
	        vector<float> seg = quiz[i][j];
	        createBoxes(1, PxVec3(2 * dim * seg[0] + (i - 0.5) * spacing, 2 * dim * seg[1] + spacing, 2 * dim * seg[2]), PxVec3(dim, dim, dim), true, _TetrisPiece2, &tetrisObjs2, &tetrisPhysx2, &tetrisStartingPos2, &tetrisPhysxStartPos);
		}
		cube_color[0] += 0.15;
	}
	cerr << "made 4" << endl;
	// adding to root
    _root->addChild(_mainTetrisSwitch2);
    _mainTetrisSwitch2->addChild(_mainTetris2);
    cerr << "added" << endl;
	// reset the color
	cube_color = Vec4(1, 1, 1, 1); cube_alpha = 1.0;
}

void SpatialViz::createTetrisPiece(int size, int numPieces, vector<Vec3> &cubeLocations) {
    // size = dimension of the cube we want the tetris piece to fit in
    // numPieces = number of small cubes to make up the tetris piece
    // cubeLocations is the vector to store the center of the locations of the cubes
    
    // Initiallize and setup cube for maze generation
	int*** cube = new int**[size];
	for (int i = 0; i < size; i++) {
		
		cube[i] = new int*[size];
		for (int j = 0; j < size; j++) {
			
			cube[i][j] = new int[size];
			for (int k = 0; k < size; k++) {
			    
			    int temp = rand() % (999);
			    cube[i][j][k] = temp + 1;       // values range from 1 to 1000 (0 means visited)
			}
		}
	}
    
    // PRINT OUT THE MATRIX
	/*for (int currLayer = 0; currLayer < size; currLayer++){
	    cerr << "layer " << currLayer << endl;
	    for (int x = 0; x < size; x++) {
	        for (int y = 0; y < size; y++) {
	            cerr << setw(2) << cube[x][y][currLayer] << " : " ;
	        }cerr << endl;
	    }cerr << endl << endl;
	}*/
    
    // Setup for iteration of prim's algorithm
	priority_queue< pair<int, Vec3 > > q = priority_queue< pair<int, Vec3 > >   ();

    // mark the first corner as visited
	cube[0][0][0] = 0;
	
	// defining the position to pass into the queue
	Vec3 pos1 = Vec3( 1, 0, 0 );
	Vec3 pos2 = Vec3( 0, 1, 0 );
	Vec3 pos3 = Vec3( 0, 0, 1 );
	
		
	// adding the potential wall to the queue
	//                 value        position of wall
	q.push( make_pair(cube[1][0][0], pos1));    // right
	q.push( make_pair(cube[0][1][0], pos2));    // down
	q.push( make_pair(cube[0][0][1], pos3));    // up a level
    
    // accumulate the cube locations
    cubeLocations.push_back(Vec3(0,0,0));
    while (!q.empty() && cubeLocations.size() < numPieces) {

		// Look at the top(greatest) element
		pair<int, Vec3 > edge = q.top();
		q.pop();
		
		//cerr << "popped: " << edge.second[0] << " : " << edge.second[1] << " : " << edge.second[2] << ", pVal = " << edge.first << endl;
		        
        int x = edge.second[0];
		int y = edge.second[1];
		int z = edge.second[2];
		
		// if you have not yet visited that location 
		if (cube[x][y][z] != 0) {
		
		    // mark as visited
		    cube[x][y][z] = 0;
		    
		    // add the cube to cubeLocations
		    cubeLocations.push_back(Vec3(x,y,z));
		    
		    // push the neighbors into the queue
		    // right and left neighbors
		    if (x-1 >= 0 && cube[x-1][y][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x-1][y][z], Vec3(x-1, y, z)));
		    }
		    if (x+1 < size && cube[x+1][y][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x+1][y][z], Vec3(x+1, y, z)));
		    }
		    // up and down neighbors
		    if (y-1 >= 0 && cube[x][y-1][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y-1][z], Vec3(x, y-1, z)));
		    }
		    if (y+1 < size && cube[x][y+1][z]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y+1][z], Vec3(x, y+1, z)));
		    }
		    // upstairs and downstairs neighbors
		    if (z-1 >= 0 && cube[x][y][z-1]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y][z-1], Vec3(x, y, z-1)));
		    }
		    if (z+1 < size && cube[x][y][z+1]) {
		        //cube[x][y][z] = 0;
		        q.push(make_pair(cube[x][y][z+1], Vec3(x, y, z+1)));
		    }
		}
    }
    
    // delete cube
    for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			delete(cube[i][j]);
		}delete(cube[i]);
	}delete(cube);
}

void SpatialViz::createPuzzleCube(int size) {
    if (size < 2) return;

    int cleanliness = 50;   // 0 - 100
		int straightener = 50;  // 0 - charMax
		srand((unsigned)time(0));

		// Initiallize and setup cube for maze generation
		char*** cube = new char**[size];
		for (int i = 0; i < size; i++) {
			cube[i] = new char*[size];
			for (int j = 0; j < size; j++) {
				cube[i][j] = new char[size];
				for (int k = 0; k < size; k++) {
					if ((i == 0 || j == 0 || k == 0 || i == size - 1 || j == size - 1 || k == size - 1)) {
						cube[i][j][k] = rand() % (CHAR_MAX - 1) + 2;
					} else {
						cube[i][j][k] = 1;
					}
				}
			}
		}
		
		// Setup for iteration of prim's algorithm
		priority_queue< pair<char, Vec3 > > q = priority_queue< pair<char, Vec3 > > ();
        
		cube[0][0][0] = 0;
		// defining the position to pass into the queue
	    Vec3 pos1 = Vec3( 1, 0, 0 );
	    Vec3 pos2 = Vec3( 0, 1, 0 );
	    Vec3 pos3 = Vec3( 0, 0, 1 );
		q.push(make_pair(cube[1][0][0], pos1));
		q.push(make_pair(cube[0][1][0], pos2));
		q.push(make_pair(cube[0][0][1], pos3));

        while (!q.empty()) {

			// Look at the top(greatest) element
			pair<char, Vec3 > edge = q.top();
			q.pop();

			int x = edge.second[0];
			int y = edge.second[1];
			int z = edge.second[2];
			
			// Check if pixel creates a loop
			int direction = 1;
			if (x > 0 && cube[x - 1][y][z] == 0) direction *= 10;
			if (x < size - 1 && cube[x + 1][y][z] == 0) direction *= 60;
			if (y > 0 && cube[x][y - 1][z] == 0) direction *= 20;
			if (y < size - 1 && cube[x][y + 1][z] == 0) direction *= 50;
			if (z > 0 && cube[x][y][z - 1] == 0) direction *= 30;
			if (z < size - 1 && cube[x][y][z + 1] == 0) direction *= 40;

			if (direction > 60) continue;
			
			// Set pixel to visited
			cube[x][y][z] = 0;

			// Push neighbors, while adjusting weights so that straight is encouraged
			if (x > 0 && cube[x - 1][y][z] > 1) {
				Vec3 temp = Vec3(x - 1, y, z);
				if (direction == 60) cube[x - 1][y][z] = (cube[x - 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x - 1][y][z] + straightener;
				q.push(make_pair(cube[x - 1][y][z], temp));
			}
			if (x < size - 1 && cube[x + 1][y][z] > 1) {
				Vec3 temp = Vec3( x + 1, y, z);
				if (direction == 10) cube[x + 1][y][z] = (cube[x + 1][y][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x + 1][y][z] + straightener;
				q.push(make_pair(cube[x + 1][y][z], temp));
			}
			if (y > 0 && cube[x][y - 1][z] > 1) {
				Vec3 temp = Vec3( x, y - 1, z);
				if (direction == 50) cube[x][y - 1][z] = (cube[x][y - 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y - 1][z] + straightener;
				q.push(make_pair(cube[x][y - 1][z], temp));
			}
			if (y < size - 1 && cube[x][y + 1][z] > 1) {
				Vec3 temp = Vec3( x, y + 1, z );
				if (direction == 20) cube[x][y + 1][z] = (cube[x][y + 1][z] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y + 1][z] + straightener;
				q.push(make_pair(cube[x][y + 1][z], temp));
			}
			if (z > 0 && cube[x][y][z - 1] > 1) {
				Vec3 temp = Vec3( x, y, z - 1 );
				if (direction == 40) cube[x][y][z - 1] = (cube[x][y][z - 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z - 1] + straightener;
				q.push(make_pair(cube[x][y][z - 1], temp));
			}
			if (z < size - 1 && cube[x][y][z + 1] > 1) {
				Vec3 temp = Vec3( x, y, z + 1 );
				if (direction == 30) cube[x][y][z + 1] = (cube[x][y][z + 1] > CHAR_MAX - straightener) ? CHAR_MAX : cube[x][y][z + 1] + straightener;
				q.push(make_pair(cube[x][y][z + 1], temp));
			}
	    }
	    
	    // Clean up pointless dead ends
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				for (int z = 0; z < size; z++) {
					if ((x == 0 || y == 0 || z == 0 || x == size - 1
						|| y == size - 1 || z == size - 1) && cube[x][y][z] == 0) {

						int direction = 1;
						if (x > 0 && cube[x - 1][y][z] == 0) direction *= 10;
						if (x < size - 1 && cube[x + 1][y][z] == 0) direction *= 60;
						if (y > 0 && cube[x][y - 1][z] == 0) direction *= 20;
						if (y < size - 1 && cube[x][y + 1][z] == 0) direction *= 50;
						if (z > 0 && cube[x][y][z - 1] == 0) direction *= 30;
						if (z < size - 1 && cube[x][y][z + 1] == 0) direction *= 40;

						if (direction > 60) continue;

						if (rand() % 100 > cleanliness) {
							int count = 0;

							switch (direction) {
							case 10:
								if (y > 0 && cube[x - 1][y - 1][z] == 0) count++;
								if (y < size - 1 && cube[x - 1][y + 1][z] == 0) count++;
								if (z > 0 && cube[x - 1][y][z - 1] == 0) count++;
								if (z < size - 1 && cube[x - 1][y][z + 1] == 0) count++;

								break;
							case 60:
								if (y > 0 && cube[x + 1][y - 1][z] == 0) count++;
								if (y < size - 1 && cube[x + 1][y + 1][z] == 0) count++;
								if (z > 0 && cube[x + 1][y][z - 1] == 0) count++;
								if (z < size - 1 && cube[x + 1][y][z + 1] == 0) count++;

								break;
							case 20:
								if (x > 0 && cube[x - 1][y - 1][z] == 0) count++;
								if (x < size - 1 && cube[x + 1][y - 1][z] == 0) count++;
								if (z > 0 && cube[x][y - 1][z - 1] == 0) count++;
								if (z < size - 1 && cube[x][y - 1][z + 1] == 0) count++;

								break;
							case 50:
								if (x > 0 && cube[x - 1][y + 1][z] == 0) count++;
								if (x < size - 1 && cube[x + 1][y + 1][z] == 0) count++;
								if (z > 0 && cube[x][y + 1][z - 1] == 0) count++;
								if (z < size - 1 && cube[x][y + 1][z + 1] == 0) count++;

								break;
							case 30:
								if (x > 0 && cube[x - 1][y][z - 1] == 0) count++;
								if (x < size - 1 && cube[x + 1][y][z - 1] == 0) count++;
								if (y > 0 && cube[x][y - 1][z - 1] == 0) count++;
								if (y < size - 1 && cube[x][y + 1][z - 1] == 0) count++;

								break;
							case 40:
								if (x > 0 && cube[x - 1][y][z + 1] == 0) count++;
								if (x < size - 1 && cube[x + 1][y][z + 1] == 0) count++;
								if (y > 0 && cube[x][y - 1][z + 1] == 0) count++;
								if (y < size - 1 && cube[x][y + 1][z + 1] == 0) count++;

								break;
							default:
								break;
							}

							if (count > 1) cube[x][y][z] = 2;

						}

					}
				}
			}
		}
		
		// Load walls
		vector<Vec3> positions = vector<Vec3>();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < size; k++) {
					if ( (i == 0 || j == 0 || k == 0 || i == size - 1 || j == size - 1 || k == size - 1) && cube[i][j][k] > 1) {
		                Vec3 temp( (float)i, (float)j, (float)k );
						positions.push_back(temp);
					} 
				}
				delete(cube[i][j]);
			}
			delete(cube[i]);
		}
		delete(cube);
		
		
		// draw the cubes
		currScene = gScene2;
		//cerr << "PUZZLE CUBE - size of positions = " << positions.size() << endl;
		cube_color = Vec4(1.0, 0.2, 0.2, 1.0);
		for (int i = 0; i < positions.size(); i++) {
		    Vec3 pos = positions[i]*0.05;
		    createBoxes(1, PxVec3(pos[0], pos[1], pos[2]), PxVec3(0.025, 0.025, 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        }
        cube_color = Vec4(1, 0.6, 0.2, 1.0);
        createBoxes(1, PxVec3(0.025 * (size - 1), 0.025 * (size - 1), 0.025 * (size - 1)), PxVec3(0.025 * (size - 2), 0.025 * (size - 2), 0.025 * (size - 2)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        cube_color = Vec4(1,1,1,1);
        
        cube_alpha = 0.1;
        createBoxes(1, PxVec3(0.025 * (2*size), 0.025 * size - 0.025, 0.025 * size - 0.025), PxVec3(0.025, 0.025 * (size), 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        createBoxes(1, PxVec3(0.025 * (-2), 0.025 * size - 0.025, 0.025 * size - 0.025), PxVec3(0.025, 0.025 * (size), 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * (2*size), 0.025 * size - 0.025), PxVec3(0.025 * (size), 0.025, 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * (-2), 0.025 * size - 0.025), PxVec3(0.025 * (size), 0.025, 0.025 * (size)), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * size - 0.025, 0.025 * (2*size)), PxVec3(0.025 * (size), 0.025 * (size), 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        createBoxes(1, PxVec3(0.025 * size - 0.025, 0.025 * size - 0.025, 0.025 * (-2)), PxVec3(0.025 * (size), 0.025 * (size), 0.025), true, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
        
        cube_alpha = 1.0;
        // place sphere on the top level 
        createSpheres(1, PxVec3(0, 0, 0), 0.023, _objGroupMaze, &mazeObjs, &mazePhysx, &mazeStartingPos, &mazePhysxStartPos);
}


void SpatialViz::create5x5(int puzzleSize)
{
    float puzzleDim = 0.25;
    float height = 0.05;    float radius = 0.0375;
    float width = 0.001;
    float spacing = 0.1;                        // to have the maze an even 5x5 grid
    PxVec3 position = PxVec3(0.0f, 0.0f, 0.0f);
    
    
    //int puzzleSize = 5;           int layers = 2;
    int prioritySize = 2 * (puzzleSize - 1);
    
    // Initiallize and setup cube for maze generation
	int*** cube = new int**[prioritySize+1];
	for (int i = 0; i <= prioritySize; i++) {
		
		cube[i] = new int*[prioritySize+1];
		for (int j = 0; j <= prioritySize; j++) {
			
			cube[i][j] = new int[prioritySize+1];
			for (int k = 0; k <= prioritySize; k++) {
			    
			    // give the spots that represent grid locations 0 (meaning they have not yet been visited)
			    if ((i % 2 == 0 && j % 2 == 0 && k % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
					cube[i][j][k] = 0;
				}
				//  k = even => wall layer          
			    else if (k % 2 == 0) {
				    int temp = rand() % (100);
				    cube[i][j][k] = temp;
				}
				// k = odd => floor layer   i and j are even (grid locations)
				else if ( (k % 2 == 1) && (i % 2 == 0 && j % 2 == 0) ){
				    int temp = rand() % (20);
				    cube[i][j][k] = temp;
				}
				else {
				    cube[i][j][k] = 0;
				}
			}
		}
	}
	
	// PRINT OUT THE MATRIX
	/*for (int currLayer = 0; currLayer <= prioritySize; currLayer++){
	    cerr << "layer " << currLayer << endl;
	    for (int x = 0; x <= prioritySize; x++) {
	        for (int y = 0; y <= prioritySize; y++) {
	            cerr << setw(2) << cube[x][y][currLayer] << " : " ;
	        }cerr << endl;
	    }cerr << endl << endl;
	}*/
   
    // Setup for iteration of prim's algorithm
	priority_queue< pair<int, Vec3 > > q = priority_queue< pair<int, Vec3 > >   ();

    // mark the first corner as visited
	cube[0][0][0] = 1;
	
	// defining the position to pass into the queue
	Vec3 pos1 = Vec3( 1, 0, 0 );
	Vec3 pos2 = Vec3( 0, 1, 0 );
	Vec3 pos3 = Vec3( 0, 0, 1 );
	
		
	// adding the potential wall to the queue
	//                 value        position of wall
	q.push( make_pair(cube[1][0][0], pos1));    // right
	q.push( make_pair(cube[0][1][0], pos2));    // down
	q.push( make_pair(cube[0][0][1], pos3));    // up a level
    
    
    vector<Vec3> verticalWalls;
    vector<Vec3> horizontalWalls;
    vector<Vec3> floors;
    bool verticalWall, horizontalWall, floor;
    
    while (!q.empty()) {

		// Look at the top(greatest) element
		pair<int, Vec3 > edge = q.top();
		q.pop();
		
		//cerr << "popped: " << edge.second[0] << " : " << edge.second[1] << " : " << edge.second[2] << ", pVal = " << edge.first << endl;
		        
        // Check if 
		//      the node we are trying to get to has already been visited or not
		//      a boundry check
		// based on this 
		//      -> increment the appropiate x, y or z value to the next block you are trying to connect 
		
		
		
		int x = edge.second[0];
		int y = edge.second[1];
		int z = edge.second[2];
		
		// get the two grid locations 
		Vec3 grid1, grid2;  	bool visit1, visit2;
		
		// store the wall location to potentially add to the walls vector to draw
		Vec3 wallLocation;
		
		// if the row is odd, the col is even, and z is even ... horizontal wall
		if (x % 2 == 1 && y % 2 == 0 && z % 2 == 0) {
            //cerr << "\t-> horizontal wall" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << y/2 << ", " << x/2+1 << endl;
            horizontalWall = true;          verticalWall = false;       floor = false;
            
            grid1 = grid2 = edge.second;
            grid1[0]--;
            grid2[0]++;
            
            // store the indices for drawing the walls 
            wallLocation[0] = y/2;      // l
            wallLocation[1] = x/2+1;    // w
            wallLocation[2] = z/2;      // level
            
        }
        // if the row is even, the col is odd, and z is even ... vertical wall 
		else if (x % 2 == 0 && y % 2 == 1 && z % 2 == 0) {
		    //cerr << "\t-> vertical wall" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << x/2 << ", " << y/2+1 << endl;
		    horizontalWall = false;          verticalWall = true;       floor = false;
		    
		    grid1 = grid2 = edge.second;
		    grid1[1]--;
		    grid2[1]++;
		    
		    // store the indices for drawing the walls 
            wallLocation[0] = x/2;      // l
            wallLocation[1] = y/2+1;    // w
            wallLocation[2] = z/2;      // level
           
		}
		// if x and y (row/col) are even, and z is odd ... floor tile 
		else if (x % 2 == 0 && y % 2 == 0 && z % 2 == 1) {
		    //cerr << "\t-> floor" << " (x,y) = " << x << ", " << y << " -> (l, w) = " << x/2 << ", " << y/2 << " level = " << z/2+1 << endl;
		    horizontalWall = false;          verticalWall = false;       floor = true;
		    
		    grid1 = grid2 = edge.second;
		    grid1[2]--;
		    grid2[2]++;
		    
		    // store the indices for drawing the floor
		    wallLocation[0] = x/2;      // l
		    wallLocation[1] = y/2;      // w
		    wallLocation[2] = z/2+1;    // level
		}
		else continue;
		
		// have we visited the two grid locations?
		visit1 = (int)cube[(int)grid1[0]][(int)grid1[1]][(int)grid1[2]];
		visit2 = (int)cube[(int)grid2[0]][(int)grid2[1]][(int)grid2[2]];
		
		//cerr << "\t-> connecting: " << endl;
        //cerr << "\t" << grid1[0] << " : " << grid1[1] << " : " << grid1[2] << ", visited? " << visit1 << endl;
        //cerr << "\t" << grid2[0] << " : " << grid2[1] << " : " << grid2[2] << ", visited? " << visit2 << endl;


        // check if we both of those grid locations are occupied or not
        if (!( visit1 && visit2)){
            // if we have not visited both the grid locations yet, knock down the wall(s) and mark them as visited
            
            // push on the neighbors for the unvisited node and mark that node as visited
            int x,y,z;              // the grid location (+- 1 is the wall location +-2 is the neighboring grid location
            
            // if we haven't visited grid1
            if (!visit1){
                
                // mark grid1 as visited
                cube[(int)grid1[0]][(int)grid1[1]][(int)grid1[2]] = 1;
                
                x = (int)grid1[0];
                y = (int)grid1[1];
                z = (int)grid1[2];
                
                // push on grid1's neighbors
                // go up/down
                // if we haven't visited the up neighbor
                if (x-2 >= 0 && !cube[x-2][y][z]) {
                   Vec3 temp = Vec3(x-1, y, z);
                    q.push(make_pair(cube[x-1][y][z], temp ));
                }
                // if we haven't visited the down neighbor
                if ( x+2 <= prioritySize && !cube[x+2][y][z]) {
                    Vec3 temp = Vec3(x+1, y, z);
                    q.push(make_pair(cube[x+1][y][z], temp ));
                }
                
                // go left/right
                // if we haven't visited the left neighbor
                if (y-2 >= 0 && !cube[x][y-2][z]) {
                    Vec3 temp = Vec3(x, y-1, z);
                    q.push(make_pair(cube[x][y-1][z], temp ));
                }
                // if we haven't visited the right neighbor
                if (y+2 <= prioritySize && !cube[x][y+2][z]) {
                    Vec3 temp = Vec3(x, y+1, z);
                    q.push(make_pair(cube[x][y+1][z], temp ));
                }
                
                // go upstairs/downstairs
                // if we haven't visited the upstairs neighbor
                if (z - 2 >= 0 && !cube[x][y][z-2] ) {
                    Vec3 temp = Vec3(x, y, z-1);
                    q.push( make_pair( cube[x][y][z-1], temp ) );
                }
                
                // if we haven't visited the upstairs neighbor
                if( z+2 <= prioritySize && !cube[x][y][z+2]) {
                    Vec3 temp = Vec3(x, y, z+1);
                    q.push( make_pair( cube[x][y][z+1], temp ) );
                }
                
            }
            // if we haven't visited grid2
            if (!visit2) {
                
                // mark grid2 as visited
                cube[(int)grid2[0]][(int)grid2[1]][(int)grid2[2]] = 1;
                
                // push on grid2's neighbors
                x = (int)grid2[0];
                y = (int)grid2[1];
                z = (int)grid2[2];
                
                // push on grid1's neighbors
                
                // go up/down
                // if we haven't visited the up neighbor 
                if (x - 2 >= 0 && !cube[x-2][y][z]) {
                    Vec3 temp = Vec3(x-1, y, z);
                    q.push(make_pair(cube[x-1][y][z], temp ));
                }
                // if we haven't visited the down neighbor
                if (x + 2 <= prioritySize && !cube[x+2][y][z]) {
                    Vec3 temp = Vec3(x+1, y, z);
                    q.push(make_pair(cube[x+1][y][z], temp ));
                }
                
                // go left/right
                // if we haven't visited the left neighbor
                if (y-2 >= 0 && !cube[x][y-2][z]) {
                    Vec3 temp = Vec3(x, y-1, z);
                    q.push(make_pair(cube[x][y-1][z], temp ));
                }
               // if we haven't visited the right neighbor
                if (y+2 <= prioritySize && !cube[x][y+2][z]) {
                    Vec3 temp = Vec3(x, y+1, z);
                    q.push(make_pair(cube[x][y+1][z], temp ));
                }
                
                // go upstairs/downstairs
                // if we haven't visited the upstairs neighbor
                if (z - 2 >= 0 && !cube[x][y][z-2] ) {
                    Vec3 temp = Vec3(x, y, z-1);
                    q.push( make_pair( cube[x][y][z-1], temp ) );
                }
                
               // if we haven't visited the upstairs neighbor
                if( z+2 <= prioritySize && !cube[x][y][z+2]) {
                    Vec3 temp = Vec3(x, y, z+1);
                    q.push( make_pair( cube[x][y][z+1], temp ) );
                }
            }
        }
        else {
            //cerr << "Both have been visited" << endl;
            if (verticalWall) {
                verticalWalls.push_back(wallLocation);
            }
            if (horizontalWall) {
                horizontalWalls.push_back(wallLocation);
            }
            if (floor) {
                floors.push_back(wallLocation);
            }
        }
    }
    
    // draw the puzzle
    currScene = gScene;
    cerr << "Drawing the puzzle" << endl;
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
    
    // deleteCube
    for (int i = 0; i <= prioritySize; i++) {
		for (int j = 0; j <= prioritySize; j++) {
			delete(cube[i][j]);
		}delete(cube[i]);
	}delete(cube);
    
}

void SpatialViz::createLabyrinth(float boxHeight, float floorHeight)
{      
    currScene = gScene;
    // create the frame 
    createBoxes(1, PxVec3(0.0f, floorHeight+boxHeight, .20f), PxVec3(0.15, boxHeight*2, 0.005), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(0.0f, floorHeight+boxHeight, 0.0f), PxVec3(0.15, boxHeight*2, 0.005), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(-0.155f, floorHeight+boxHeight, 0.1f), PxVec3(0.005, boxHeight*2, 0.105), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(0.155f, floorHeight+boxHeight, 0.1f), PxVec3(0.005, boxHeight*2, 0.105), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);

    // maze portions
    // left to right - horizontal boxes
    createBoxes(1, PxVec3(-0.135, floorHeight, 0.16), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(-0.135, floorHeight, 0.08), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos); 
    
    createBoxes(1, PxVec3(-0.105, floorHeight, 0.12), PxVec3(0.0225, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(-0.105, floorHeight, 0.04), PxVec3(0.0225, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(-0.055, floorHeight, 0.16), PxVec3(0.0325, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(-0.02, floorHeight, 0.12), PxVec3(0.03, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(-0.02, floorHeight, 0.08), PxVec3(0.03, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.07, floorHeight, 0.04), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(0.11, floorHeight, 0.08), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.05, floorHeight, 0.08), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(0.09, floorHeight, 0.155), PxVec3(0.03, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.13, floorHeight, 0.12), PxVec3(0.02, boxHeight, 0.0015), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    
    // top down - vertical boxes
    createBoxes(1, PxVec3(-0.025, floorHeight, 0.175), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(0.09, floorHeight, 0.175), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(-0.085, floorHeight, 0.14), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    createBoxes(1, PxVec3(-0.085, floorHeight, 0.06), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(-0.055, floorHeight, 0.02), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(-0.02, floorHeight, 0.06), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.025, floorHeight, 0.02), PxVec3(0.0015, boxHeight, 0.0175), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.0325, floorHeight, 0.13), PxVec3(0.001, boxHeight, 0.0475), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    createBoxes(1, PxVec3(0.0875, floorHeight, 0.07), PxVec3(0.0015, boxHeight, 0.0275), true, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
    
    // the sphere for the labyrinth
    createSpheres(1, PxVec3(0.125, -0.245, 0.19), 0.0075, _labyrinthGroup, &labyrinthObjs, &labyrinthPhysx, &labyrinthStartingPos, &labyrinthPhysxStartPos);
}

void SpatialViz::createBoxes(int num, PxVec3 startVec, PxVec3 dimensions, bool fixed, Group* parentGroup, vector<PositionAttitudeTransform*>* currSG, vector<PxRigidDynamic*>* currPhysX, vector<Vec3>* currStart, vector<PxVec3>* currStartPhysx, PxQuat quat) 
{
    // set the density, dimenstions, and material properties
    PxReal density = 1.0;
    PxBoxGeometry geometryBox(dimensions); 
    PxMaterial* mMaterial = mPhysics->createMaterial(0.1,0.2,0.5);;
    
    
    for (int i = 0; i < num; i++)
    {
        //cerr << "----- Creating a cube ----- " << endl;
        
        PxVec3 currVec = PxVec3(startVec.x, startVec.y+(i*0.12), startVec.z);
        //PxTransform transform(currVec, PxQuat::createIdentity());
        PxTransform transform(currVec, quat);
        
        bool ans = quat == PxQuat::createIdentity();
        if (!ans) cerr << "NOT IDENTITY" << endl;

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
        ///gScene->addActor(*actor);    
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
        //gScene->addActor(*actor);                                       
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
    //static bool firstNodeOn = true;
    //static bool secondNodeOn = true;

    if (menuItem == _puzzle1Button)
    {
        cerr << "Maze Puzzle Button pressed" << endl;
        // shows the first puzzle only (maze)
        _root->setSingleChildOn(5); // was 0
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(0);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        
        currScene = gScene2;
        
        currSG = &mazeObjs;
        currPhysx = &mazePhysx;
        currStartingPositions = &mazeStartingPos;
        currPhysxStartPos = &mazePhysxStartPos;
        
        restartPhysics();
    }
    if (menuItem == _puzzle2Button)
    {
        cerr << "5x5 Puzzle Button pressed" << endl;  
        // shows the second puzzle only (5x5)
        _root->setSingleChildOn(4); // was 1
        
        _5x5Switch->setSingleChildOn(0);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1); 
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        
        currScene = gScene;
        
        currSG = &fiveObjs;
        currPhysx = &fivePhysx;
        currStartingPositions = &fiveStartingPos;
        currPhysxStartPos = &fivePhysxStartPos;
        
        restartPhysics();
    }
    if (menuItem == _puzzle3Button)
    {
        cerr << "Cube Puzzle Button pressed" << endl;  
        // shows the third puzzle only (5 piece)
        // NOW THE SECOND TETRIS OPTION
        
        _root->setSingleChildOn(2);
        
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
    if (menuItem == _labyrinthPuzzle)
    {
        cerr << "Labyrinth Puzzle Button pressed" << endl;
        _root->setSingleChildOn(3);
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(0); 
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        
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
        _root->setSingleChildOn(6);
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(0);
        _mainTetrisSwitch->setSingleChildOn(0);
        
        // currScene = gSceneTetris;
        currSG = &tetrisObjs;
        currPhysx = &tetrisPhysx;
        currStartingPositions = &tetrisStartingPos;
        currPhysxStartPos = &tetrisPhysxStartPos;
        
        restartPhysics();
    }
    
    if (menuItem == _removePuzzles)
    {
        cerr << "Remove Puzzles " << endl;  
        // shows no puzzles
        _root->setSingleChildOn(8);   
        
        _5x5Switch->setSingleChildOn(1);
        _mazeSwitch->setSingleChildOn(1);
        _labyrinthSwitch->setSingleChildOn(1);  
        _tetrisSwitch->setSingleChildOn(1);
        _mainTetrisSwitch->setSingleChildOn(1);
        
        _tetrisSwitch2->setSingleChildOn(1);
        _mainTetrisSwitch2->setSingleChildOn(1);
        
        
        // to change later
        //currSG = &mazeObjs;
        //currPhysx = &mazePhysx;
        //currStartingPositions = &mazeStartingPos;
        //currPhysxStartPos = &mazePhysxStartPos;
    }
    if (menuItem == _restartPhysics)
    {
        cerr << "restart Physics" << endl;
        restartPhysics();
    }
  
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

    _puzzle1Button = new MenuButton("Maze Puzzle");
    _puzzle1Button->setCallback(this);
    _mainMenu->addItem(_puzzle1Button);
    
	_puzzle2Button = new MenuButton("5x5 Puzzle");
    _puzzle2Button->setCallback(this);
    _mainMenu->addItem(_puzzle2Button);
    
    _puzzle3Button = new MenuButton("Cube Puzzle");
    _puzzle3Button->setCallback(this);
    _mainMenu->addItem(_puzzle3Button);
    
    _labyrinthPuzzle = new MenuButton("Labyrinth Puzzle");
    _labyrinthPuzzle->setCallback(this);
    _mainMenu->addItem(_labyrinthPuzzle);
    
    _tetrisPuzzle = new MenuButton("Tetris Matching");
    _tetrisPuzzle->setCallback(this);
    _mainMenu->addItem(_tetrisPuzzle);
    
    _removePuzzles = new MenuButton("Remove Puzzles");
    _removePuzzles->setCallback(this);
    _mainMenu->addItem(_removePuzzles);
        
	_restartPhysics = new MenuButton("Restart Physics");
    _restartPhysics->setCallback(this);
    _mainMenu->addItem(_restartPhysics);
	
	// --------------- create the root "Node" ---------------
	_root = new Switch();		
    
    
    // make a Pyramid and add it to the root 
    //Geode * pyramidGeode = makePyramid(Vec3(0,0,0), Vec3(250, 250, 250));

	// --------------- load an image ---------------
	//Texture2D * faceTexture = loadTexture("/home/klucknav/Downloads/index.tga");
	
	// --------------- add that image to the state ---------------
	//StateSet * stateOne = new StateSet();
	//stateOne->setTextureAttributeAndModes(0, faceTexture, StateAttribute::ON);

 
    // --------------- Load the Puzzles ---------------
    _puzzleMazeGroup = new Group;
    _puzzle5x5Group = new Group;
    _piecePuzzleGroup = new Group;
    _labyrinthGroup = new Group;
    _labyrinthSwitch = new Switch;
    
    Vec3 scale = Vec3(25, 25, 25);
    
    // set up model path:
    string configPath = ConfigManager::getEntry("Plugin.SpatialViz.DataDir");
    cerr << configPath << endl;
    
    // PuzzleMaze and add to the puzzleMazeGroup       Scale      transform        alpha
	_puzzleMaze = loadOBJ(_puzzleMazeGroup, configPath+PATH_MAZE, scale, Vec3(-150, 0.0, 0.0), 1.0f);   // name, scale, trans, alpha
	_mazeBox = loadOBJ(_puzzleMaze, configPath+PATH_MAZE_BOX, Vec3(1,1,1), Vec3(0,0,0), 0.5f);         // name, scale, trans, alpha


    // 5x5 Puzzle and add to the puzzle5x5Group
    _puzzle5x5 = loadOBJ(_puzzle5x5Group, configPath+PATH_5X5, scale, Vec3(-150, 0.0, 0.0), 0.5f);  // name, scale, trans, alpha
    
    
    // 5 Piece Puzzle
    _piecePuzzle1 = loadOBJ(_piecePuzzleGroup, configPath+PATH_PUZZLE1, scale, Vec3(-150, 0.0, 0.0), 1.0f);    // name, scale, trans, alpha
    _piecePuzzle2 = loadOBJ(_piecePuzzleGroup, configPath+PATH_PUZZLE2, scale, Vec3(-75, 250, 0.0), 1.0f);    // name, scale, trans, alpha
    _piecePuzzle3 = loadOBJ(_piecePuzzleGroup, configPath+PATH_PUZZLE3, scale, Vec3(-300, 0.0, 0.0), 1.0f);    // name, scale, trans, alpha
    _piecePuzzle4 = loadOBJ(_piecePuzzleGroup, configPath+PATH_PUZZLE4, scale, Vec3(-50, -250, 0.0), 1.0f);   // name, scale, trans, alpha
    _piecePuzzle5 = loadOBJ(_piecePuzzleGroup, configPath+PATH_PUZZLE5, scale, Vec3(100, 0.0, 0.0), 1.0f);     // name, scale, trans, alpha
    
    
    // add puzzle models to _root
    _root->addChild(_puzzleMazeGroup);
    _root->addChild(_puzzle5x5Group);
    _root->addChild(_piecePuzzleGroup);
    _root->addChild(_labyrinthSwitch);
    _labyrinthSwitch->addChild(_labyrinthGroup);
    
    
    
    // ------------------------ PhysX -------------------
    
    // _objGroup for testing with PhysX boxes/spheres
    _5x5Switch = new Switch;
	_objGroup = new Group;
	_root->addChild(_5x5Switch);
	_5x5Switch->addChild(_objGroup); 
	
	_mazeSwitch = new Switch;
	_objGroupMaze = new Group;
	_root->addChild(_mazeSwitch);
	_mazeSwitch->addChild(_objGroupMaze);
	
	_tetrisSwitch = new Switch;
	_objGroupTetris = new Group;
	_root->addChild(_tetrisSwitch);
	
	_tetrisSwitch2 = new Switch;
	_TetrisPiece2 = new Group;
	_root->addChild(_tetrisSwitch2);
	_tetrisSwitch2->addChild(_TetrisPiece2);
	
	initPhysX();
	
 	// from Points 		-------------------------------------------------------
 	//                              name, navigation, movable, clip, context menu, show bounds
	SceneObject *soLab = new SceneObject("root", true, false, true, true, false);
	PluginHelper::registerSceneObject(soLab,"SpatialVizObject");
	//so->addChild(_root);
	soLab->addChild(_labyrinthSwitch);				// adding the _root to the scene
	soLab->attachToScene();				
	soLab->addMoveMenuItem();
	soLab->addNavigationMenuItem();
	cerr << "added the root to the scene " << endl;
	//                              name, navigation, movable, clip, context menu, show bounds
	SceneObject *so5x5 = new SceneObject("5x5", true, false, true, true, false);
	PluginHelper::registerSceneObject(so5x5,"SpatialVizObject");
	so5x5->addChild(_5x5Switch);				// adding the _root to the scene
	so5x5->attachToScene();				
	so5x5->addMoveMenuItem();
	so5x5->addNavigationMenuItem();
	//                              name, navigation, movable, clip, context menu, show bounds
	SceneObject *soMaze = new SceneObject("Maze", true, false, true, true, false);
	PluginHelper::registerSceneObject(soMaze,"SpatialVizObject");
	soMaze->addChild(_mazeSwitch);				// adding the _root to the scene
	soMaze->attachToScene();				
	soMaze->addMoveMenuItem();
	soMaze->addNavigationMenuItem();
	//                                      name, navigation, movable, clip, context menu, show bounds
	SceneObject *soTetris = new SceneObject("tetris4", false, false, true, true, false);
	PluginHelper::registerSceneObject(soTetris,"SpatialVizObject");
	soTetris->addChild(_tetrisSwitch);				// adding the _root to the scene
	soTetris->attachToScene();				
	soTetris->addMoveMenuItem();
	soTetris->addNavigationMenuItem();
	//                                            name, navigation, movable, clip, context menu, show bounds
	SceneObject *soMainTetris = new SceneObject("mainTetris", true, true, true, true, false);
	PluginHelper::registerSceneObject(soMainTetris,"SpatialVizObject");
	soMainTetris->addChild(_mainTetrisSwitch);				// adding the _root to the scene
	soMainTetris->attachToScene();				
	soMainTetris->addMoveMenuItem();
	soMainTetris->addNavigationMenuItem();
    
    //                                      name, navigation, movable, clip, context menu, show bounds
	SceneObject *soTetris2 = new SceneObject("tetris4_2", false, false, true, true, false);
	PluginHelper::registerSceneObject(soTetris2,"SpatialVizObject");
	soTetris2->addChild(_tetrisSwitch2);				// adding the _root to the scene
	soTetris2->attachToScene();				
	soTetris2->addMoveMenuItem();
	soTetris2->addNavigationMenuItem();
	//                                            name, navigation, movable, clip, context menu, show bounds
	SceneObject *soMainTetris2 = new SceneObject("mainTetris_2", true, true, true, true, false);
	PluginHelper::registerSceneObject(soMainTetris2,"SpatialVizObject");
	soMainTetris2->addChild(_mainTetrisSwitch2);				// adding the _root to the scene
	soMainTetris2->attachToScene();				
	soMainTetris2->addMoveMenuItem();
	soMainTetris2->addNavigationMenuItem();
		
	
	/*cerr << "adding the next so" << endl;
	// testing 		-------------------------------------------------------
	SceneObject *so2 = new SceneObject("tetris test", true, false, false, true, false);
	PluginHelper::registerSceneObject(so2,"SpatialVizObject");
	so2->addChild(_TetrisPiece);				// adding the _root to the scene
	so2->attachToScene();				
	so2->setNavigationOn(true);
	so2->addMoveMenuItem();
	so2->addNavigationMenuItem();
	so2->setShowBounds(false); 
	so2->setMovable(true);*/
	
	
	
	// show no puzzles
    _root->setSingleChildOn(8);
    
    _5x5Switch->setSingleChildOn(1);
    _mazeSwitch->setSingleChildOn(1);
    _labyrinthSwitch->setSingleChildOn(1);  
    _tetrisSwitch->setSingleChildOn(1);
    _mainTetrisSwitch->setSingleChildOn(1);
    _tetrisSwitch2->setSingleChildOn(1);
    _mainTetrisSwitch2->setSingleChildOn(1);
    return true;
}



// load in an OBJ file
PositionAttitudeTransform * SpatialViz::loadOBJ(Group * parent, string filename, Vec3 scale, Vec3 trans, float alpha)
{
    // LOAD IN A OBJ FILE
	osg::Node* objNode = osgDB::readNodeFile(filename);
	osg::PositionAttitudeTransform *modelScale = new PositionAttitudeTransform;
	
	if (objNode==NULL)
	{ 
	    std::cerr << "SpatialViz: Error reading file " << filename << endl;
	}	
	else if (objNode != NULL)
	{
	    setNodeTransparency(objNode, alpha);
	
	    modelScale->setScale(scale);
	    modelScale->setPosition(trans);
	
	    modelScale->addChild(objNode);
	    parent->addChild(modelScale);
	}
	//return objNode;
    return modelScale;	
}


Texture2D * SpatialViz::loadTexture(string filename)
{
    // load an image:
	Texture2D * faceTexture = new Texture2D;
	faceTexture->setDataVariance(Object::DYNAMIC);
	
	Image * faceIm = osgDB::readImageFile(filename);
	if (!faceIm) {
		cerr << "couldn't find texture" << filename << endl;
	}
	
	faceTexture->setImage(faceIm);

	StateSet * stateOne = new StateSet();
	stateOne->setTextureAttributeAndModes(0, faceTexture, StateAttribute::ON);
	
	return faceTexture;
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

// make a Pyramid with the given transform information and attach to the root node
Geode * SpatialViz::makePyramid(Vec3 pos, Vec3 scale)
{
    // make a Pyramid
	Geode * pyramidGeode = new Geode();		        // a geometry node to collect drawables 
	Geometry * pyramidGeometry = new Geometry();	// Geometry instance to associate the 
							                        // vertices and vertex data 

	pyramidGeode->addDrawable(pyramidGeometry);	    // attaching the geometry to the drawable

	// the vertex data
	Vec3Array * pyramidVertices = new Vec3Array;
	pyramidVertices->push_back(Vec3(0, 0, 0));
	pyramidVertices->push_back(Vec3(2, 0, 0));
	pyramidVertices->push_back(Vec3(2, 2, 0));
	pyramidVertices->push_back(Vec3(0, 2, 0));
	pyramidVertices->push_back(Vec3(1, 1, 2));

	pyramidGeometry->setVertexArray(pyramidVertices);

	// NOTES:
	// addPrimitiveSet takes a primitiveSet type 
	// pyramidBase needs to be a primitiveSet type BUT since PrimitiveSet is an abstract class 
	// we need to use a class that inherits PrimitiveSet... 
	// DrawElements is ... NOPE still abstract
	//
	// PrimitiveSet test = new PrimitiveSet(DrawElements); // PyramidSet = abstract
	// DrawElements * pyramidBase = new DrawElements(PrimitiveSet::QUADS, 0, 0); 
	// DrawElements = abstract

	// Basically creating the EBO's for each face
	// Base	
	DrawElementsUInt * pyramidBase = new DrawElementsUInt(PrimitiveSet::QUADS);
	pyramidBase->push_back(3);
	pyramidBase->push_back(2);
	pyramidBase->push_back(1);
	pyramidBase->push_back(0);
	pyramidGeometry->addPrimitiveSet(pyramidBase);
	
	// Faces:
	DrawElementsUInt * pyramidFace1 = new DrawElementsUInt(PrimitiveSet::TRIANGLES);
	pyramidFace1->push_back(0);
	pyramidFace1->push_back(1);
	pyramidFace1->push_back(4);
	pyramidGeometry->addPrimitiveSet(pyramidFace1);

	DrawElementsUInt * pyramidFace2 = new DrawElementsUInt(PrimitiveSet::TRIANGLES);
	pyramidFace2->push_back(1);
	pyramidFace2->push_back(2);
	pyramidFace2->push_back(4);
	pyramidGeometry->addPrimitiveSet(pyramidFace2);

	DrawElementsUInt * pyramidFace3 = new DrawElementsUInt(PrimitiveSet::TRIANGLES);
	pyramidFace3->push_back(2);
	pyramidFace3->push_back(3);
	pyramidFace3->push_back(4);
	pyramidGeometry->addPrimitiveSet(pyramidFace3);
	
	DrawElementsUInt * pyramidFace4 = new DrawElementsUInt(PrimitiveSet::TRIANGLES);
	pyramidFace4->push_back(3);
	pyramidFace4->push_back(0);
	pyramidFace4->push_back(4);
	pyramidGeometry->addPrimitiveSet(pyramidFace4);

	// Array of colors 
	Vec4Array * colors = new Vec4Array;
	colors->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f));	// index 1: red
	colors->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f));	// index 2: green
	colors->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f));	// index 3: blue
	colors->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f));	// index 4: white

	// assign colors to vertices
	TemplateIndexArray<unsigned int, Array::UIntArrayType,4,4> *colorIndexArray;
	colorIndexArray = new TemplateIndexArray<unsigned int, Array::UIntArrayType, 4, 4>;
	
	colorIndexArray->push_back(0); 		// vertex 0 assigned to color index 0 
	colorIndexArray->push_back(1); 		// vertex 1 assigned to color index 1 
	colorIndexArray->push_back(2); 		// vertex 2 assigned to color index 2 
	colorIndexArray->push_back(3); 		// vertex 3 assigned to color index 3 
	colorIndexArray->push_back(0); 		// vertex 4 assigned to color index 0

	// associate the color array with the actual geometry
	pyramidGeometry->setColorArray(colors);				// attach the actual colors
	// NOTE: says that Geometry does not have this function.. deprecated 
	//pyramidGeometry->setColorIndices(colorIndexArray);		// attach the array that says
									// which color goes with which vertex
	pyramidGeometry->setColorBinding(Geometry::BIND_PER_VERTEX);	// since we are attaching a 
									// color to each vertex 
	
	// texture coordinates	
	Vec2Array * texCoords = new Vec2Array(5);
	(*texCoords)[0].set(0.00f, 0.0f);
	(*texCoords)[1].set(0.25f, 0.0f);
	(*texCoords)[2].set(0.50f, 0.0f);
	(*texCoords)[3].set(0.75f, 0.0f);
	(*texCoords)[4].set(0.50f, 1.0f);
	
	pyramidGeometry->setTexCoordArray(0, texCoords);
	
	// Declare and initialize a transform Nodoe
	PositionAttitudeTransform * pyramid2xForm = new PositionAttitudeTransform;

	// position/scale the model
	pyramid2xForm->setPosition(pos);
	pyramid2xForm->setScale(scale);
	
	_root->addChild(pyramid2xForm);
	pyramid2xForm->addChild(pyramidGeode);
	
	return pyramidGeode;
	
}

// this is the draw callback, gets called every frame
void SpatialViz::preFrame()
{
    //if( currTime < end) 
	{
	    //cerr << "----- in preFrame ----- \n";
	    currScene->simulate(myTimestep);   // advance the simulation by myTimestep
	    currScene->fetchResults();

        // ---------- Update gravity vector: ----------
        //osg::Matrix w2o = PluginHelper::getObjectToWorldTransform();
        //osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
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
        
        _sceneDesc->gravity=PxVec3(gravity[0], gravity[1], gravity[2]);
        currScene->setGravity(PxVec3(gravity[0], gravity[1], gravity[2]));
        
        //cerr << "gravity=" << gravity[0] << ", " << gravity[1] << ", " << gravity[2] << endl;  
        if (*currSG == tetrisObjs){
            cerr << "TETRIS" << endl;
            cerr << "size of currSG = " << currSG->size() << endl;
            // need to check the orientation of the last 5 cubes with any of the first 4 puzzles
            for (int i = currSG->size(); i >= 0; i--){
                
            }
        }
        
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

