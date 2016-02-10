#ifndef _PHYSICSLAB_H
#define _PHYSICSLAB_H

#include <utility>
#include <vector>
#include <stack>
#include <string>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/Navigation.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrKernel/InteractionEvent.h>

// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/io_utils>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/MatrixTransform>
#include <osg/Node>

#include <OASClient.h>  // audio server

// Local
#include "ObjectFactory.h"

enum KEYS {
  W = 119,
  A = 97,
  S = 115,
  D = 100,
  SPACE = 32
};

class PhysicsLab : public cvr::CVRPlugin, public cvr::MenuCallback
{
  protected:
    cvr::SubMenu * _mainMenu, * _loadMenu, *_toolBoxMenu;
    cvr::MenuButton *_removeButton;
    cvr::MenuButton *_loadScene, *_resetScene;
    cvr::MenuButton *_ramp0Spawn, *_cubeSpawn, *_ringSpawn, *_rimSpawn;
    bool haveAudio;
    oasclient::Sound* soundCollision;
    oasclient::Sound* soundAmbient;
    oasclient::Sound* soundApplause;

  public:
    PhysicsLab();
    virtual ~PhysicsLab();
    void menuCallback(cvr::MenuItem*);
    bool init();
    void preFrame();
    bool processEvent(cvr::InteractionEvent*);
    void resetScene();
    void setupScene();
    
  private:
    std::string datadir;
    ObjectFactory * of;
    bool sceneLoaded;
    std::vector< std::pair<osg::MatrixTransform*, osg::Matrixd> > gen_objects;
    int numSpheres;
    std::vector<MatrixTransform*> spheres;
    float minDistance, maxDistance;
    float joyY, joyX;
    
    void addToScene(osg::MatrixTransform*);
    void initSound();

    bool wonGame;
    std::vector< Vec4 > colorArray;
    osg::Matrix resetTrans;
};

#endif
