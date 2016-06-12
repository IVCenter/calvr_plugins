#include "PhysicsLab.h"

#include <PluginMessageType.h>
#include <cvrConfig/ConfigManager.h>

#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

// OSG:
#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Vec3d>

#define NUM_SPHERES 20

using namespace std;
using namespace cvr;
using namespace osg;

CVRPLUGIN(PhysicsLab)

void setupScene( ObjectFactory* );

const double pi = 3.141593;
int frame = 0;
MatrixTransform * camNode, *boxMatrix, *seesawMatrix;
PositionAttitudeTransform * lightMat;
StateSet* lightSS;
int boxId, seesawid;
std::vector<int> sphereId;
std::vector<MatrixTransform*> sphereMatrix;
MatrixTransform* handBall;

// Nav
bool goLeft = false, goRight = false, goUp = false, goDown = false;

// Constructor
PhysicsLab::PhysicsLab()
{
}

void PhysicsLab::menuCallback(MenuItem* menuItem)
{
  if (menuItem == _loadScene && !sceneLoaded) {
    setupScene();
    sceneLoaded = true;
  } else if (sceneLoaded) {
    if (menuItem == _resetScene)
      resetScene();
    else if (menuItem == _ramp0Spawn)
      camNode->addChild(of->addCustomObject( datadir + "/objects/ramp.osgb", 100.0f, Vec3(0,0,100), Quat(0, Vec3(0,1,0)), false, true ));
    else if (menuItem == _cubeSpawn)
      camNode->addChild(of->addCustomObject( datadir + "/objects/cube.wrl", 2.0f, Vec3(0,0,100), Quat(), false, true ));
    else if (menuItem == _ringSpawn)
      camNode->addChild(of->addCustomObject( datadir + "/objects/ring.osgb", 110.0f, Vec3(0,0,100), Quat(), false, true ));
    else if (menuItem == _rimSpawn)
      camNode->addChild(of->addCustomObject( datadir + "/objects/halfring.osgb", 70.0f, Vec3(0,0,100), Quat(), false, true ));
  }
}

// intialize
bool PhysicsLab::init()
{
  cerr << "PhysicsLab::PhysicsLab" << endl;
  sceneLoaded = false;
  numSpheres = 0;
  camNode = NULL;
  wonGame = false;

  // add ball colors
  colorArray.push_back(Vec4(1.0,1.0,1.0,1.0));
  colorArray.push_back(Vec4(1.0,0.0,0.0,1.0));
  colorArray.push_back(Vec4(0.0,1.0,0.0,1.0));
  colorArray.push_back(Vec4(0.0,0.0,1.0,1.0));
  colorArray.push_back(Vec4(1.0,0.0,1.0,1.0));
  colorArray.push_back(Vec4(1.0,1.0,0.0,1.0));
  colorArray.push_back(Vec4(0.0,1.0,1.0,1.0));
 
  // reset starting position
  resetTrans.setTrans(Vec3(1900,1900,500));
  
  joyX =joyY = 0.0;  
  minDistance = ConfigManager::getFloat("Plugin.PhysicsLab.MinDistance", 800.0);
  maxDistance = ConfigManager::getFloat("Plugin.PhysicsLab.MaxDistance", 3000.0);
  
  _mainMenu = new SubMenu("PhysicsLab", "PhysicsLab");
  _mainMenu->setCallback(this);
  MenuSystem::instance()->addMenuItem(_mainMenu);
  
  _loadScene = new cvr::MenuButton("Load Scene");
  _mainMenu->addItem(_loadScene);
  _loadScene->setCallback(this);
  
  _resetScene = new cvr::MenuButton("Reset Scene");
  _mainMenu->addItem(_resetScene);
  _resetScene->setCallback(this);
  
  _toolBoxMenu = new SubMenu("Toolbox", "Toolbox");
  _toolBoxMenu->setCallback(this);
  _mainMenu->addItem(_toolBoxMenu);
  
  _ramp0Spawn = new cvr::MenuButton("Ramp");
  _toolBoxMenu->addItem(_ramp0Spawn);
  _ramp0Spawn->setCallback(this);
  
  _cubeSpawn = new cvr::MenuButton("Cube");
  _toolBoxMenu->addItem(_cubeSpawn);
  _cubeSpawn->setCallback(this);
  
  _ringSpawn = new cvr::MenuButton("Ring");
  _toolBoxMenu->addItem(_ringSpawn);
  _ringSpawn->setCallback(this);
  
  _rimSpawn = new cvr::MenuButton("Half Ring");
  _toolBoxMenu->addItem(_rimSpawn);
  _rimSpawn->setCallback(this);
  
  of = new ObjectFactory();
  
  // init sound:
  haveAudio = false;
  soundCollision = NULL;
  soundAmbient = NULL;
  soundApplause = NULL;
  
  return true;
}

void PhysicsLab::initSound()
{
  string oas_ip = ConfigManager::getEntry("Plugin.PhysicsLab.OASip");
  int oas_port = ConfigManager::getInt("value", "Plugin.PhysicsLab.OASport", 31231);
  if (oasclient::ClientInterface::initialize(oas_ip, oas_port))
  {
   haveAudio = true;
   cerr << "Connection to OAS sound server established." << endl;

   // Create a collision sound using the file specified by filepath
   string soundFileCollision = ConfigManager::getEntry("Plugin.PhysicsLab.CollisionSound");
   soundCollision = new oasclient::Sound(soundFileCollision);
   if (soundCollision->isValid())
   {
     cerr << "Created the collision sound at " << soundFileCollision << endl;
   }
   else
   {
     cerr << "Unable to create the collision sound at " << soundFileCollision << endl;
   }

   // Create a collision sound using the file specified by filepath
   string soundFileAmbient = ConfigManager::getEntry("Plugin.PhysicsLab.AmbientSound");
   soundAmbient = new oasclient::Sound(soundFileAmbient);
   if (soundAmbient->isValid())
   {
     cerr << "Created the ambient sound at " << soundFileAmbient << endl;
     soundAmbient->setPosition(0, 0, 0);
     soundAmbient->setDirection(0, 0, 0);
     soundAmbient->setLoop(true);
     soundAmbient->play();
   }
   else
   {
     cerr << "Unable to create the ambient sound at " << soundFileAmbient << endl;
   }
   
   // Create applause sound:
   string soundFileApplause = ConfigManager::getEntry("Plugin.PhysicsLab.ApplauseSound");
   soundApplause = new oasclient::Sound(soundFileApplause);
   if (soundApplause->isValid())
   {
     cerr << "Created the applause sound at " << soundFileApplause << endl;
     soundApplause->setPosition(0, 0, 0);
     soundApplause->setDirection(0, 0, 0);
     soundApplause->setLoop(false);  
   }
   else
   {
     cerr << "Unable to create the applause sound at " << soundFileApplause << endl;
   }
  }
  else
  {
   cerr << "Warning: unable to create a connection with the sound server. Disabling audio." << endl;
   haveAudio = false;
  }
}

void PhysicsLab::addToScene( osg::MatrixTransform* node ) {
  Matrixd m = node->getMatrix();
  gen_objects.push_back(std::make_pair(node, m));
  
  if (camNode) camNode->addChild(node);
}

void PhysicsLab::setupScene( ) 
{
    
    float x, y, z;
    //x = ConfigManager::getFloat("x", "Plugin.PhysicsLab.PlayAreaPosition", 0.0);
    x = 0.0; // dont want to set in the cave
    y = ConfigManager::getFloat("y", "Plugin.PhysicsLab.PlayAreaPosition", 0.0);
    z = ConfigManager::getFloat("z", "Plugin.PhysicsLab.PlayAreaPosition", 0.0);
    
    // adjust y based on min distance
    if( y < minDistance )
        y = minDistance;
    else if( y > maxDistance )
        y = maxDistance;
   
    // root node 
    camNode = new MatrixTransform;
	  
    Matrix cam2;
    cam2.makeTranslate(x,y,z);
    camNode->setMatrix(cam2);
    PluginHelper::getScene()->addChild( camNode );
    
    // bounding box (invis)
    //of->addHollowBox( Vec3(0,0,0), Vec3(2000,2000,5000), false, false );
    
    // World Gravity, Elevator, then Funnel
    of->addAntiGravityField( Vec3(0,0,0), Vec3(10000,10000,10000), Vec3(0,0,-5000), true );
    camNode->addChild( of->addAntiGravityField( Vec3(-1850,0,700), Vec3(100,100,650), Vec3(0,0,500), false ) );

    // goal bucket
    of->addGoalZone( Vec3(1850, 0, 200), Vec3(80, 80, 20) );
    
    datadir = ConfigManager::getEntry("Plugin.PhysicsLab.DataDir");
		
    if (datadir.empty()) 
    {
        cout << "PhysicsLab: Could not load data directory." << endl;
    } 
    else 
    {
	// AGF shroud
        camNode->addChild( of->addCustomObject( datadir + "/objects/tower.osgb", 30.f, Vec3(-1960,-125,110), Quat(0, Vec3(0,1,0)), false, false, 0.0f ) );
    
        // two ramps
        addToScene(of->addCustomObject( datadir + "/objects/ramp.osgb", 115.0f, Vec3(-500,-100,300), Quat(90.f * pi / 180.f, Vec3(0,0,1)) * Quat(25.f * pi / 180.f, Vec3(0,1,0)), false, true ));
        addToScene(of->addCustomObject( datadir + "/objects/ramp.osgb", 100.0f, Vec3(-750,-1250,100), Quat(90.f * pi / 180.f, Vec3(0,0,1)), false, true ));
        
        // add a tramp
        addToScene(of->addCustomObject( datadir + "/objects/tramp.osgb", 100.0f, Vec3(-750,1250,100), Quat(90.f * pi / 180.f, Vec3(0,0,1)), false, true, 2.0f ));
        
        // two pipes
        addToScene(of->addCustomObject( datadir + "/objects/pipesolid.osgb", 100.0f, Vec3(-1700,1700,100), Quat(), false, true ));
        addToScene(of->addCustomObject( datadir + "/objects/pipesolid.osgb", 60.0f, Vec3(-1000,1700,200), Quat(90.f * pi / 180.f, Vec3(0,1,0)), false, true ));
    
        addToScene(of->addCustomObject( datadir + "/objects/plate.osgb", 100.0f, Vec3(400,1400,50), Quat(90.f * pi / 180.f, Vec3(0,0,1)), false, true ));
        addToScene(of->addCustomObject( datadir + "/objects/wedge.osgb", 100.0f, Vec3(1000,-1700,50), Quat(), false, true ));
    
        addToScene(of->addCustomObject( datadir + "/objects/halfring.osgb", 100.0f, Vec3(-1500,-1700,50), Quat(), false, true ));
        addToScene(of->addCustomObject( datadir + "/objects/halfring.osgb", 70.0f, Vec3(0,-1700,50), Quat(90.f * pi / 160.f, Vec3(1,0,0)), false, true ));
    
        // what is going on here?? it is moveable or not??
        //camNode->addChild(of->addCustomObject( datadir + "/objects/ring.osgb", 110.0f, Vec3(1000,1700,200), Quat(90.f * pi / 180.f, Vec3(1,0,0)), false, true ));
        addToScene(of->addCustomObject( datadir + "/objects/ring.osgb", 110.0f, Vec3(1000,1700,200), Quat(90.f * pi / 180.f, Vec3(1,0,0)), false, true ));
    
        // goal ring
        camNode->addChild(of->addCustomObject( datadir + "/objects/ring.osgb", 110.0f, Vec3(1850,0,200), Quat(), false, false ));
	  
        // open box (pit)
        //MatrixTransform* openpit = of->addOpenBox( Vec3(0,0,101), Vec3(2000,2000,100), 50.0, false, true );
        MatrixTransform* openpit = of->addCustomObject( datadir + "/objects/pit.osgb", 1.f / 39.f, Vec3(0,0,0), Quat(0, Vec3(0,1,0)), false, false );
        camNode->addChild( openpit );
    }
     
    Group* skybox = of->addSkybox(datadir + "/objects/skybox/");\
    MatrixTransform* skyboxScale = new MatrixTransform();
    skyboxScale->setMatrix(Matrixd::scale(200000.f,200000.f,200000.f));
    skyboxScale->addChild(skybox);
    camNode->addChild(skyboxScale);

    // initalize sound
    if(cvr::ComController::instance()->isMaster()) 
        initSound();
}

void PhysicsLab::resetScene() {
  for (int i = 0; i < gen_objects.size(); ++i) {
    of->setWorldTransform(gen_objects[i].first, gen_objects[i].second);
  }

  of->resetGame();
  wonGame = false;
}

// this is called if the plugin is removed at runtime
PhysicsLab::~PhysicsLab()
{
    delete of;
}

void PhysicsLab::preFrame()
{
  if (!sceneLoaded) return;
  
    // Camera Movement
    if (goLeft || goRight || goUp || goDown) {
      const float diff = 3.f;
      float xdiff = diff * PluginHelper::getLastFrameDuration() * 300.0f;
      
      Matrix cam = camNode->getMatrix();
      if (goLeft && !goRight) {
        cam.setTrans( cam.getTrans() + cam.getRotate()*Vec3(xdiff,0.f,0.f) );
      } else if (goRight && !goLeft) {
        cam.setTrans( cam.getTrans() + cam.getRotate()*Vec3(-xdiff,0.f,0.f) );
      }
      
      if (goUp && !goDown) {
        cam.setTrans( cam.getTrans() + cam.getRotate()*Vec3(0.f,0.f,-xdiff) );
      } else if (goDown && !goUp) {
        cam.setTrans( cam.getTrans() + cam.getRotate()*Vec3(0.f,0.f,xdiff) );
      }
      
      camNode->setMatrix(cam);
	  }
	  
    frame = (frame + 1) % 30000;
    static bool startSim = false;
    if (frame == 120) {
      startSim = true;
    }
    
    //check if ball is below the pit
    std::vector<osg::MatrixTransform* >::iterator it = spheres.begin();
    for( ; it != spheres.end(); ++it)
    {
        // if below table remove
        if((*it)->getMatrix().getTrans().z() < -100.0 )
        {
          of->setWorldTransform((*it), resetTrans);
          of->setLinearVelocity((*it), Vec3(0,0,0));
        }
    }
    
    
    if (frame % 120 == 0 && startSim && numSpheres < NUM_SPHERES) {
      //MatrixTransform* sphereMat = of->addSphere( Vec3(-1900,0,200), 35, colorArray[numSpheres%4], true, true );
      MatrixTransform* sphereMat = of->addSphere( Vec3(1900,1900,500), 35, colorArray.at(numSpheres%colorArray.size()), true, true );
      //addToScene(sphereMat);
      
      spheres.push_back(sphereMat);
      //spheres.push(sphereMat);
      //Matrixd sphere_m;
      //gen_objects.push_back(std::pair<MatrixTransform*, Matrixd>(sphereMat, sphere_m));
      camNode->addChild( sphereMat );
      numSpheres++;
    }
    
    Matrixd handmat = PluginHelper::getHandMat(0);
    of->updateHand(handmat, camNode->getMatrix());
    
    if (frame % 1500 == 0) {
      std::cout << "FPS: " << 1.0/PluginHelper::getLastFrameDuration() << std::endl;
    }
    
    if (of->wonGame() && !wonGame) 
    {
    
      //////

      // hack to keep playing sounds when balls go in
      //wonGame = true;
      
      // resets the winding state so it will immediately be able to trigger the winning sound
      of->resetGame();
      
      //////

      if (haveAudio && soundApplause) 
          soundApplause->play();
    }
      
    if (startSim) of->stepSim( PluginHelper::getLastFrameDuration() );
}

bool PhysicsLab::processEvent(InteractionEvent * event) 
{
    
    // only process events if scene is loaded
    if( !sceneLoaded )
        return false;
    
    static bool grabbing = false;
    
    KeyboardInteractionEvent * kp;
    TrackedButtonInteractionEvent * he;
    ValuatorInteractionEvent * vie;
    
    if ((kp = event->asKeyboardEvent()) != NULL) {
        if (kp->getInteraction() == KEY_DOWN) {
          switch (kp->getKey()) {
              case A:
                  goLeft = true;
                  break;
              case D:
                  goRight = true;
                  break;
              case W:
                  goUp = true;
                  break;
              case S:
                  goDown = true;
                  break;
              case SPACE:
                  break;
          }
        } else if (kp->getInteraction() == KEY_UP) {
          switch (kp->getKey()) {
              case A:
                  goLeft = false;
                  break;
              case D:
                  goRight = false;
                  break;
              case W:
                  goUp = false;
                  break;
              case S:
                  goDown = false;
                  break;
              case SPACE:
                  break;
          }
      }
    } else if ((he = event->asTrackedButtonEvent()) != NULL) {
      if (he->getHand() == 0 && he->getButton() == 0) {
          Matrixd handmat = PluginHelper::getHandMat(0);
        if (he->getInteraction() == BUTTON_DOWN && !grabbing)
            grabbing = of->grabObject( handmat, PluginHelper::getScene() );
        else if (he->getInteraction() == BUTTON_UP) {
            grabbing = false;
            of->releaseObject();
        }
      }
    } else if((vie = event->asValuatorEvent()) != NULL) {

        bool move = false;
        bool rotate = false;

        osg::Matrix rotation;
        osg::Vec3 moveVec;

        // do navigation around the box
        if( vie->getValuator() == 1)
        {
            joyY = vie->getValue();
                                   
            if( abs(joyX) *.5 >= abs(vie->getValue()) )
                vie->setValue(0);
                                                                   
            moveVec.y() = vie->getValue() * fabs(vie->getValue()) * 70.0f;
            move = true;
        }
        else
        {
            joyX = vie->getValue();
            if( abs(joyY) *.5 > abs(vie->getValue()) )
                vie->setValue(0);

            float angle = vie->getValue() * fabs(vie->getValue()) * M_PI * (1.0 / 120.0);

            osg::Matrix r;
            r.makeRotate(-angle,osg::Vec3(0,0,1));
            rotation = rotation * r;
            rotate= true;
        }

        if( camNode )
        {
            osg::Matrix mat = camNode->getMatrix();

            if( rotate )
            {
                // shift to origin to rotate
                osg::Vec3 originOffset = mat.getTrans();
                mat = mat * osg::Matrix::translate(-originOffset) * rotation
                          * osg::Matrix::translate(originOffset);

            }

            if( move )
            {
                // add difference
                float yValue = -moveVec.y() + mat.getTrans().y();

                if( yValue < minDistance)
                    mat.setTrans(osg::Vec3(0.0, minDistance, mat.getTrans().z()));
                else if ( yValue > maxDistance )
                    mat.setTrans(osg::Vec3(0.0, maxDistance, mat.getTrans().z()));
                else
                    mat.setTrans(osg::Vec3(0.0, yValue, mat.getTrans().z()));
            }

            if( rotate || move )
                camNode->setMatrix(mat);

        }
    }


    return true;
}
