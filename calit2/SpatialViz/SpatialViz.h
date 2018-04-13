#ifndef _SPATIALVIZ_H
#define _SPATIALVIZ_H

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
#include <GL/freeglut.h>                    // testing - PhysX3

// PhysX:
#include <PxPhysicsAPI.h>
#include <extensions/PxDefaultErrorCallback.h>
#include <extensions/PxDefaultAllocator.h>
#include <foundation/Px.h>
#include <extensions/PxShapeExt.h>
#include <foundation/PxMat33.h>



class SpatialViz : public cvr::CVRPlugin, public cvr::MenuCallback
{
  protected:
    // menu variables
    cvr::SubMenu *_mainMenu;
    cvr::MenuButton *_puzzle1Button, *_puzzle2Button, *_puzzle3Button, *_labyrinthPuzzle, *_removePuzzles, *_restartPhysics;
    
    osg::PositionAttitudeTransform *_sphereTrans, *_cubeTrans;
    osg::Geode *_cubeGeode, *_sphereGeode;
    
    osg::Switch *_root;
    
    // Puzzle variables
    osg::PositionAttitudeTransform * _puzzleMaze, *_mazeBox, *_puzzle5x5, * _piecePuzzle1, * _piecePuzzle2, * _piecePuzzle3, * _piecePuzzle4, * _piecePuzzle5;
    osg::Group * _puzzleMazeGroup, *_puzzle5x5Group, *_piecePuzzleGroup;
    osg::Group * _objGroup;
    
    std::vector<osg::Vec3> startingPositions;
    std::vector<physx::PxVec3> physxStartPos;
    
    physx::PxSceneDesc* _sceneDesc;
    
    // functions
    osg::PositionAttitudeTransform *loadOBJ(osg::Group *, std::string, osg::Vec3, osg::Vec3, float);
    //osg::PositionAttitudeTransform *loadOBJ(osg::PositionAttitudeTransform *, std::string, osg::Vec3, osg::Vec3, float);
    
    void setNodeTransparency(osg::Node*, float);
    osg::Geode * makePyramid(osg::Vec3, osg::Vec3);
    osg::Texture2D * loadTexture(std::string);
    osg::PositionAttitudeTransform  * addSphere(osg::Group*, osg::Vec3, float, osg::Vec3);
    osg::PositionAttitudeTransform  * addCube(osg::Group*, osg::Vec3, float, float, float, osg::Vec3);
 
  public:
    SpatialViz();
    virtual ~SpatialViz();
    
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
    
    // PhysX 
    void initPhysX();
    void createLabyrinth(float, float);
    void createBoxes(int, physx::PxVec3, physx::PxVec3, bool);
    void createSpheres(int, physx::PxVec3, float);
    bool advance(physx::PxReal);

    
};

#endif
