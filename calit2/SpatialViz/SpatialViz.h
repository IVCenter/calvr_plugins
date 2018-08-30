#ifndef _SPATIALVIZ_H
#define _SPATIALVIZ_H

// STD
#include <queue>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>

// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

#include <osg/PositionAttitudeTransform> 	// the transform of objects 


#ifdef HAVE_PHYSX

// PhysX:
#include <PxPhysicsAPI.h>
#include <extensions/PxDefaultErrorCallback.h>
#include <extensions/PxDefaultAllocator.h>
#include <foundation/Px.h>
#include <extensions/PxShapeExt.h>
#include <foundation/PxMat33.h>
#include <foundation/PxQuat.h>

#endif


class SpatialViz : public cvr::CVRPlugin, public cvr::MenuCallback
{
  protected:
    // menu and CVR variables
    cvr::SubMenu *_mainMenu;
    cvr::MenuButton *_mazePuzzleButton, *_5x5puzzleButton, *_tetrisPuzzle2, *_labyrinthPuzzle, *_tetrisPuzzle, *_removePuzzles, *_restartPhysics;
    cvr::SceneObject *soLab, *so5x5, *soMaze, *soTetris, *soMainTetris, *soTetris2, *soMainTetris2;
    
    osg::PositionAttitudeTransform *_sphereTrans, *_cubeTrans;
    osg::Geode *_cubeGeode, *_sphereGeode;
    osg::Switch *_root, *_labyrinthSwitch, *_5x5Switch, *_mazeSwitch, *_tetrisSwitch, *_mainTetrisSwitch, *_tetrisSwitch2, *_mainTetrisSwitch2;
    
    // Puzzle variables
    osg::PositionAttitudeTransform * _puzzleMaze, *_mazeBox, *_puzzle5x5, * _piecePuzzle1, * _piecePuzzle2, * _piecePuzzle3, * _piecePuzzle4, * _piecePuzzle5;
    osg::Group * _puzzleMazeGroup, *_puzzle5x5Group, *_piecePuzzleGroup, *_labyrinthGroup;
    osg::Group * _objGroup, *_objGroupMaze, *_objGroupTetris, *_TetrisPiece, *_mainTetris, *_TetrisPiece2, *_mainTetris2;
    
#ifdef HAVE_PHYSX
    physx::PxSceneDesc* _sceneDesc;
#endif
    
    // functions
    void setNodeTransparency(osg::Node*, float);
    osg::PositionAttitudeTransform  * addSphere(osg::Group*, osg::Vec3, float, osg::Vec3);
    osg::PositionAttitudeTransform  * addCube(osg::Group*, osg::Vec3, float, float, float, osg::Vec3);
 
  public:
    SpatialViz();
    virtual ~SpatialViz();
    
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void resetSceneManager();
    void preFrame();
    
    // PhysX 
    void initPhysX();
    void restartPhysics();
    
    // puzzles
    void createTetris(int);
    void createTetris2(int);
    void createPuzzleCube(int);
    void create5x5(int);
    void createLabyrinth(float, float);

#if(PHYSX_VERSION >= 33)
    // createIdentity() and createZero() are deprecated since 3.3
    void createBoxes(int, physx::PxVec3, physx::PxVec3, bool, osg::Group*, std::vector<osg::PositionAttitudeTransform*>*, std::vector<physx::PxRigidDynamic*>*, std::vector<osg::Vec3>*, std::vector<physx::PxVec3>*, physx::PxQuat quat = physx::PxQuat(physx::PxIDENTITY()));
#else
    void createBoxes(int, physx::PxVec3, physx::PxVec3, bool, osg::Group*, std::vector<osg::PositionAttitudeTransform*>*, std::vector<physx::PxRigidDynamic*>*, std::vector<osg::Vec3>*, std::vector<physx::PxVec3>*, physx::PxQuat quat = physx::PxQuat::createIdentity());
#endif
    void createSpheres(int, physx::PxVec3, float, osg::Group*, std::vector<osg::PositionAttitudeTransform*>*, std::vector<physx::PxRigidDynamic*>*, std::vector<osg::Vec3>*, std::vector<physx::PxVec3>*);

    
};

#endif
