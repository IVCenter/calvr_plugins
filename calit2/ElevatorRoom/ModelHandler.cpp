#include "ModelHandler.h"


using namespace cvr;
using namespace osg;
using namespace std;

namespace ElevatorRoom
{

#define DING_OFFSET 1
#define EXPLOSION_OFFSET 9
#define LASER_OFFSET 17

ModelHandler::ModelHandler()
{
    _audioHandler = NULL;
    _activeObject = NULL;
    _geoRoot = new osg::MatrixTransform();
    _crosshairPat = NULL;
    _scoreText = NULL;
    _scoreSwitch = NULL;

    _dataDir = ConfigManager::getEntry("Plugin.ElevatorRoom.DataDir");
    _dataDir = _dataDir + "/";

    _loaded = false;
    _doorDist = 0;
    _activeDoor = 0;
    _viewedDoor = 4;
    _lightColor = 0;
    _doorInView = false;
    _totalAngle = 0;

    _colors.push_back(osg::Vec4(1, 1, 1, 1));   // WHITE
    _colors.push_back(osg::Vec4(1, 0, 0, 1));   // RED
    _colors.push_back(osg::Vec4(0, 0, 1, 1));   // BLUE
    _colors.push_back(osg::Vec4(1, 0.5, 0, 1)); // ORANGE
    _colors.push_back(osg::Vec4(1, 1, 0, 1));   // YELLOW
    _colors.push_back(osg::Vec4(0, 1, 0, 1));   // GREEN
    _colors.push_back(osg::Vec4(0.3, 0.15, 0.0, 1.0)); // BROWN
    _colors.push_back(osg::Vec4(0.7, 0.7, 0.7, 1.0));  // GREY

    _wallTex = ConfigManager::getEntry("Plugin.ElevatorRoom.WallTexture");
    _floorTex = ConfigManager::getEntry("Plugin.ElevatorRoom.FloorTexture");
    _ceilingTex = ConfigManager::getEntry("Plugin.ElevatorRoom.CeilingTexture");
    _doorTex = ConfigManager::getEntry("Plugin.ElevatorRoom.DoorTexture");
    _elevTex = ConfigManager::getEntry("Plugin.ElevatorRoom.ElevatorTexture");

    _alienTex = ConfigManager::getEntry("Plugin.ElevatorRoom.AlienTexture");
    _allyTex = ConfigManager::getEntry("Plugin.ElevatorRoom.AllyTexture");

    _checkTex1 = ConfigManager::getEntry("Plugin.ElevatorRoom.CheckerTexture1");
    _checkTex2 = ConfigManager::getEntry("Plugin.ElevatorRoom.CheckerTexture2");
}

ModelHandler::~ModelHandler()
{

}

void ModelHandler::setAudioHandler(AudioHandler * handler)
{
    if (ComController::instance()->isMaster())
    {
        _audioHandler = handler;
    }
}

void ModelHandler::update()
{
    osg::Vec3 pos, headpos, handdir;
    headpos = PluginHelper::getHeadMat(0).getTrans();
    handdir = osg::Vec3(0,1,0) * PluginHelper::getHandMat(0);
    osg::Matrixd rotMat;
    rotMat = PluginHelper::getHandMat(0);
    rotMat.setTrans(osg::Vec3(0,0,0));
    pos = PluginHelper::getHeadMat(0).getTrans() + (osg::Vec3(-32, 200, 0) * rotMat);

    _crosshairPat->setPosition(pos); 


    float angle = 0, tolerance = 15 * (M_PI / 180), roomRad = 6;
    osg::Vec3 headForward, headToDoor;
    headForward = osg::Vec3(1, 0, 0) * PluginHelper::getObjectMatrix();

    Vec3 lightPos = osg::Quat(_activeDoor * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, roomRad, 0.0);
    headToDoor = lightPos - (osg::Vec3(0,0,0) * PluginHelper::getHeadMat());

    headForward.normalize();
    headToDoor.normalize();

    angle = acos(headForward * headToDoor);
    angle -= M_PI;
    if (0)//angle < 0) 
        angle *= -1;

    if (angle < tolerance && !_doorInView)
    {
        _doorInView = true;
        std::cout << "Door entering view" << std::endl;
    }
    else if (angle > tolerance && _doorInView)
    {
        std::cout << "Door leaving view" << std::endl;
        _doorInView = false;
    }


    if (_turningLeft)
    {
        osg::Matrix objmat = PluginHelper::getObjectMatrix();
        
        float angle = -(M_PI / 4) / 10;
        osg::Matrix turnMat;
        turnMat.makeRotate(angle, osg::Vec3(0, 0, 1));

        osg::Vec3 origin = _root->getMatrix().getTrans();

        osg::Matrix m;
        m = objmat * osg::Matrix::translate(-origin) * turnMat * 
            osg::Matrix::translate(origin);

        SceneManager::instance()->setObjectMatrix(m);

        _totalAngle += angle;
        if (_totalAngle < -M_PI / 4)
        {
            _turningLeft = false;
            _totalAngle = 0;
        }
    }
    if (_turningRight)
    {
        osg::Matrix objmat = PluginHelper::getObjectMatrix();
        
        float angle = (M_PI / 4) / 10;
        osg::Matrix turnMat;
        turnMat.makeRotate(angle, osg::Vec3(0, 0, 1));

        osg::Vec3 origin = _root->getMatrix().getTrans();

        osg::Matrix m;
        m = objmat * osg::Matrix::translate(-origin) * turnMat * 
            osg::Matrix::translate(origin);

        SceneManager::instance()->setObjectMatrix(m);

        _totalAngle += angle;
        if (_totalAngle > M_PI / 4)
        {
            _turningRight = false;
            _totalAngle = 0;
        }
    }
}

void ModelHandler::clear()
{
    _lights.clear();
    _aliensSwitch.clear();
    _alliesSwitch.clear();
    _checkersSwitch.clear();
    _lightSwitch.clear();
    _leftdoorSwitch.clear();
}

void ModelHandler::setLevel(string level)
{
    std::string tag = "Plugin.ElevatorRoom.Levels." + level;

    _wallTex = ConfigManager::getEntry(tag + ".WallTexture");
    _floorTex = ConfigManager::getEntry(tag + ".FloorTexture");
    _ceilingTex = ConfigManager::getEntry(tag + ".CeilingTexture");
    _doorTex = ConfigManager::getEntry(tag + ".DoorTexture");
    _elevTex = ConfigManager::getEntry(tag + ".ElevatorTexture");

    _alienTex = ConfigManager::getEntry(tag + ".AlienTexture");
    _allyTex = ConfigManager::getEntry(tag + ".AllyTexture");

    _checkTex1 = ConfigManager::getEntry(tag + ".CheckerTexture1");
    _checkTex2 = ConfigManager::getEntry(tag + ".CheckerTexture2");


    osg::ref_ptr<osg::Texture2D> tex;
    std::vector<osg::ref_ptr<osg::Switch> >::iterator it;
    osg::ref_ptr<osg::Image> img;

    img = osgDB::readImageFile(_dataDir + _alienTex);
    if (!img) 
    {
        std::cout << "Failed to load image " << _alienTex << "." << std::endl;
    }
    
    // Enemy
    for (it = _aliensSwitch.begin(); it != _aliensSwitch.end(); ++it)
    {
        tex = new osg::Texture2D();
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        for (int i = 0; i < (*it)->getNumChildren(); ++i)
        {
            osg::ref_ptr<osg::StateSet> state;
            state = (*it)->getChild(i)->getOrCreateStateSet();
            state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
        }
    }
    
    // Ally
    for (it = _alliesSwitch.begin(); it != _alliesSwitch.end(); ++it)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _allyTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        for (int i = 0; i < (*it)->getNumChildren(); ++i)
        {
            osg::ref_ptr<osg::StateSet> state;
            state = (*it)->getChild(i)->getOrCreateStateSet();
            state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
        }
    }
    
    std::vector<osg::ref_ptr<osg::Geode> >::iterator geoIt;

    // Walls
    for (geoIt = _walls.begin(); geoIt != _walls.end(); ++geoIt)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _wallTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        osg::ref_ptr<osg::StateSet> state;
        state = (*geoIt)->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }
    
    // Elevator doors/interior
    for (geoIt = _elevators.begin(); geoIt != _elevators.end(); ++geoIt)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _elevTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        osg::ref_ptr<osg::StateSet> state;
        state = (*geoIt)->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }
    
    // Floor
    for (geoIt = _floors.begin(); geoIt != _floors.end(); ++geoIt)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _floorTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        osg::ref_ptr<osg::StateSet> state;
        state = (*geoIt)->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }

    // Ceiling
    for (geoIt = _ceilings.begin(); geoIt != _ceilings.end(); ++geoIt)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _ceilingTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
    
        osg::ref_ptr<osg::StateSet> state;
        state = (*geoIt)->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }

    // Doors 
    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _doorTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    for (geoIt = _doors.begin(); geoIt != _doors.end(); ++geoIt)
    {
        osg::ref_ptr<osg::StateSet> state;
        state = (*geoIt)->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }
}

void ModelHandler::loadModels(osg::MatrixTransform * root)
{
    _root = root;
    if (root && _geoRoot)
    {
        root->addChild(_geoRoot.get());
    }


    float roomRad = 6.0, angle = 2 * M_PI / NUM_DOORS;

    osg::ref_ptr<osg::PositionAttitudeTransform> pat;
    osg::ref_ptr<osg::ShapeDrawable> drawable;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geo;

    osg::ref_ptr<osg::Texture2D> tex;
    osg::ref_ptr<osg::Image> img;
    osg::ref_ptr<osg::StateSet> state;

    osg::ref_ptr<osg::ShapeDrawable> redDrawable, greenDrawable, yellowDrawable,
        orangeDrawable, whiteDrawable;
    osg::ref_ptr<osg::Geode> redGeode, greenGeode, yellowGeode, orangeGeode,
        whiteGeode, blueGeode;



    // Lights
    {
    osg::ref_ptr<osg::Sphere> shape = new osg::Sphere(osg::Vec3(0,-4.75, 4.0), 0.3);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0.0, 0.0, 1.0)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));

        osg::ref_ptr<osg::Switch> switchNode;
        switchNode = new osg::Switch();
 
        for (int j = 0; j < _colors.size(); ++j)
        {
            drawable = new osg::ShapeDrawable(shape);
            drawable->setColor(_colors[j]);
            geode = new osg::Geode();
            geode->addDrawable(drawable);
            switchNode->addChild(geode, false);
        }
        switchNode->setValue(0, true);
        pat->addChild(switchNode);
        //_geoRoot->addChild(pat);

        //_lightSwitch.push_back(switchNode);
        _lights.push_back(drawable);
        
        
        // Sound
        
        osg::Vec3 pos, center, dir;
        osg::Matrix o2w, local2o;
        o2w = PluginHelper::getObjectMatrix();
        local2o = _geoRoot->getInverseMatrix();

        pos = osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0);
        pos = pos * local2o * o2w;
        center = _geoRoot->getMatrix().getTrans();
        center = center * local2o * o2w;
        dir = pos - center;

        // 1 - 8 ding sounds
        
        if (_audioHandler)
        {
            _audioHandler->loadSound(i + DING_OFFSET, i * angle);
            _audioHandler->loadSound(i + EXPLOSION_OFFSET, i * angle);
        }

    }
    }

    // Aliens 
    {
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, _colors[GREY]);
    geode->addDrawable(geo);

    osg::ref_ptr<osg::Geode> redGeode = new Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, _colors[RED]);
    redGeode->addDrawable(geo);

    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _alienTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    state = redGeode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->setScale(osg::Vec3(1, 1, 2));
        
        osg::ref_ptr<osg::Switch> switchNode;
        switchNode = new osg::Switch();
        
        switchNode->addChild(geode, false);
        switchNode->addChild(redGeode, false);
        
        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

        _aliensSwitch.push_back(switchNode);


        // Sound
        
        osg::Vec3 pos = osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0);
        osg::Vec3 dir = pos - osg::Vec3(0,0,0);

        // 9 - 16 explosion sounds
   
        if (_audioHandler)
        {
//            _audioHandler->loadSound(i + EXPLOSION_OFFSET, i * angle);
        }
   
    }   
    }

    // Allies 
    {
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, _colors[GREY]);
    geode->addDrawable(geo);
    
    redGeode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, _colors[RED]);
    redGeode->addDrawable(geo);

    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _allyTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    state = redGeode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->setScale(osg::Vec3(1, 1, 2));

        osg::ref_ptr<osg::Switch> switchNode;
        switchNode = new osg::Switch();
        
        switchNode->addChild(geode, false);
        switchNode->addChild(redGeode, false);
        
        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

        _alliesSwitch.push_back(switchNode);
    }   
    }

    // Checkerboards 
    {
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, _colors[GREY]);
    geode->addDrawable(geo);

    osg::ref_ptr<osg::Geode> geode2 = new osg::Geode();
    geode2->addDrawable(geo);

    // texture 1
    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _checkTex1);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }
    state = geode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    // texture 2
    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _checkTex2);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode2->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->setScale(osg::Vec3(1, 1, 2));

        osg::ref_ptr<osg::Switch> switchNode;
        switchNode = new osg::Switch();
        
        switchNode->addChild(geode, false);
        switchNode->addChild(geode2, false);
        
        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

        _checkersSwitch.push_back(switchNode);
    }   
    }

    // Walls
    // Elevator
    {    
    geode = new osg::Geode();
    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _wallTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    
    float wallTexScale = 0.5;

    // Left front
    geo = drawBox(osg::Vec3(3.0, -5.0, 1.0), 3.25, 0.5, 4.0, _colors[GREY], wallTexScale); 
    geode->addDrawable(geo);
    
    // Right front
    geo = drawBox(osg::Vec3(-3.0, -5.0, 1.0), 3.25, 0.5, 4.0, _colors[GREY], wallTexScale);
    geode->addDrawable(geo);

    // Top
    geo = drawBox(osg::Vec3(0.0, -5.0, 4.5), 9.0, 0.5, 3.0, _colors[GREY], wallTexScale);
    geode->addDrawable(geo);
    
    osg::ref_ptr<osg::Geode> elevatorGeode = new osg::Geode();
    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _elevTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = elevatorGeode->getOrCreateStateSet();
	state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);


    // Left side 
    geo = drawBox(osg::Vec3( 1.25, -7.0, 1.5), 0.5, 4.0, 5.0, _colors[GREY]);
    elevatorGeode->addDrawable(geo);

    // Right side 
    geo = drawBox(osg::Vec3(-1.25, -7.0, 1.5), 0.5, 4.0, 5.0, _colors[GREY]);
    elevatorGeode->addDrawable(geo);

    // Back
    geo = drawBox(osg::Vec3(0.0, -9.25, 1.5), 3.0, 0.5, 5.0, _colors[GREY]);
    elevatorGeode->addDrawable(geo);

    // Elevator floor
    geo = drawBox(osg::Vec3(0.0, -7.0, -1.15), 3.0, 3.75, 0.5, _colors[GREY]);
    elevatorGeode->addDrawable(geo);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->addChild(geode);
        pat->addChild(elevatorGeode);
        _geoRoot->addChild(pat);

        _walls.push_back(geode);
        _elevators.push_back(elevatorGeode);
    }
    }

    // Ceiling 
    { 
    geode = new osg::Geode();

    geo = drawBox(osg::Vec3(0.0, 0.0, 5.0), 40.0, 40.0, 0.1, _colors[GREY]);
    geode->addDrawable(geo);

    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _ceilingTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode->getOrCreateStateSet();
    state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

    _geoRoot->addChild(geode);
    _ceilings.push_back(geode);
    } 

    // Floor
    {
    geode = new osg::Geode();

    geo = drawBox(osg::Vec3(0.0, 0.0, -1.0), 40.0, 40.0, 0.1, _colors[GREY]);
    geode->addDrawable(geo);

    tex = new osg::Texture2D();
    img = osgDB::readImageFile(_dataDir + _floorTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }

    state = geode->getOrCreateStateSet();
    state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);   

    _geoRoot->addChild(geode);
    _floors.push_back(geode);
    }

    // Doors
    { 
    osg::ref_ptr<osg::Geometry> ldoorGeo, rdoorGeo;

    ldoorGeo = drawBox(osg::Vec3( 0.75, -5.2, 1.0), 1.5, 0.5, 4.0, _colors[GREY]);
    rdoorGeo = drawBox(osg::Vec3(-0.75, -5.2, 1.0), 1.5, 0.5, 4.0, _colors[GREY]);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _doorTex);
        if (img)
        {
            tex->setImage(img);
            tex->setResizeNonPowerOfTwoHint(false);
            tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
            tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
        }
        

        // Right door
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        _rightdoorPat.push_back(pat);
        _geoRoot->addChild(pat);
        
        osg::ref_ptr<osg::Switch> switchNode = new osg::Switch();
        osg::ref_ptr<osg::Geometry> geometry;


        /*geode = new osg::Geode();
        state = geode->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
        state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
        geode->addDrawable(rdoorGeo);
        pat->addChild(geode);*/


        for (int j = 0; j < _colors.size(); ++j)
        {
            geometry = drawBox(osg::Vec3(-0.75, -5.2, 1.0), 1.5, 0.5, 4.0, _colors[j]);
            geode = new osg::Geode();
            geode->addDrawable(geometry);

            osg::Material * mat = new osg::Material(); 
            mat->setDiffuse(Material::FRONT_AND_BACK, _colors[j]);
            mat->setSpecular(Material::FRONT_AND_BACK, _colors[j]);
            mat->setAlpha(Material::FRONT_AND_BACK, 0.5);

            state = geode->getOrCreateStateSet();
            state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
            state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
            state->setAttribute(mat, osg::StateAttribute::ON);
            state->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
            state->setRenderingHint(StateSet::TRANSPARENT_BIN);

            switchNode->addChild(geode, false);

            _doors.push_back(geode);
        }

        switchNode->setValue(GREY, true);
        _lightSwitch.push_back(switchNode);
        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

 
        // Left door
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        _leftdoorPat.push_back(pat);
        _geoRoot->addChild(pat);
        
        switchNode = new osg::Switch();

        for (int j = 0; j < _colors.size(); ++j)
        {
            geometry = drawBox(osg::Vec3(0.75, -5.2, 1.0), 1.5, 0.5, 4.0, _colors[j]);
            geode = new osg::Geode();
            geode->addDrawable(geometry);

            osg::Material * mat = new osg::Material(); 
            mat->setDiffuse(Material::FRONT_AND_BACK, _colors[j]);
            mat->setSpecular(Material::FRONT_AND_BACK, _colors[j]);
            mat->setAlpha(Material::FRONT_AND_BACK, 0.5);

            state = geode->getOrCreateStateSet();
            state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
            state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
            state->setAttribute(mat, osg::StateAttribute::ON);
            state->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
            state->setRenderingHint(StateSet::TRANSPARENT_BIN);

            switchNode->addChild(geode, false);
            _doors.push_back(geode);
        }

        switchNode->setValue(GREY, true);
        _leftdoorSwitch.push_back(switchNode);
        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

    }
    }

    // Score text
    { 
    osg::Vec3 pos = osg::Vec3(-100, 1000, -200);
    _scoreSwitch = new osg::Switch();

    _scoreText = new osgText::Text();
    _scoreText->setText("Score: 0");
    _scoreText->setCharacterSize(40);
    _scoreText->setAlignment(osgText::Text::LEFT_CENTER);
    _scoreText->setPosition(pos);
    _scoreText->setColor(osg::Vec4(1,1,1,1));
    _scoreText->setBackdropColor(osg::Vec4(0,0,0,0));
    _scoreText->setAxisAlignment(osgText::Text::XZ_PLANE);

    float width = 400, height = 100;
    osg::ref_ptr<osg::Geometry> quad = makeQuad(width, height, 
        osg::Vec4(0.8,0.8,0.8,1.0), pos - osg::Vec3(20, 0, 50));

    pat = new osg::PositionAttitudeTransform();
    geode = new osg::Geode();

    geode->addDrawable(_scoreText);
    geode->addDrawable(quad);

    geode->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    pat->addChild(geode);
  //  PluginHelper::getScene()->addChild(pat);

    _scoreSwitch->addChild(pat);
    PluginHelper::getScene()->addChild(_scoreSwitch);
    }

    // Crosshair
    {
    float width = 4;
    float height = 0.3;
    osg::Vec3 pos = osg::Vec3(0, -2500, 0);
    pos = osg::Vec3(-25, 200, 0) + PluginHelper::getHeadMat().getTrans();
    osg::Vec4 color(0.8, 0.0, 0.0, 1.0);
    osg::Geode *chGeode = new osg::Geode();
    _crosshairPat = new osg::PositionAttitudeTransform();
    _crosshairPat->setPosition(pos);
    pos = osg::Vec3(0,0,0);

    // horizontal
    osg::ref_ptr<osg::Geometry> quad = makeQuad(width, height, color, pos - osg::Vec3(width/2, 0, height/2));
    chGeode->addDrawable(quad);

    // vertical
    quad = makeQuad(height, width, color, pos - osg::Vec3(height/2, 0, width/2));
    chGeode->addDrawable(quad);

    chGeode->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
    _crosshairPat->addChild(chGeode); 

    if (ConfigManager::getEntry("Plugin.ElevatorRoom.Crosshair") == "on" )
    {
        PluginHelper::getScene()->addChild(_crosshairPat);
    }
    }

    _loaded = true;
}

osg::ref_ptr<osg::Geometry> ModelHandler::drawBox(osg::Vec3 center, float x, 
    float y, float z, osg::Vec4 color, float texScale)
{
    osg::ref_ptr<osg::Vec3Array> verts;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::Vec3Array> normals;
    osg::ref_ptr<osg::DrawArrays> primitive;
    osg::ref_ptr<osg::Geometry> geometry;
    osg::ref_ptr<osg::Vec2Array> texcoords;

    osg::Vec3 up;
    osg::Vec3 down;
    osg::Vec3 left; 
    osg::Vec3 right; 
    osg::Vec3 front; 
    osg::Vec3 back;

    float xMin = center[0] - x/2, xMax = center[0] + x/2,
          yMin = center[1] - y/2, yMax = center[1] + y/2,
          zMin = center[2] - z/2, zMax = center[2] + z/2;

    up    = osg::Vec3(0, 0,  1);
    down  = osg::Vec3(0, 0, -1);
    left  = osg::Vec3(-1, 0, 0);
    right = osg::Vec3(-1, 0, 0);
    front = osg::Vec3(0, -1, 0);
    back  = osg::Vec3(0,  1, 0);

    verts = new osg::Vec3Array(0);
    colors = new osg::Vec4Array(1);
    normals = new osg::Vec3Array(0);
    primitive = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0);
    geometry = new osg::Geometry();
    texcoords = new osg::Vec2Array(0);

    (*colors)[0] = color;

    geometry->setVertexArray(verts.get());
    geometry->setColorArray(colors.get());
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->setNormalArray(normals.get());
    geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geometry->setUseDisplayList(false);
    geometry->addPrimitiveSet(primitive.get());
    geometry->setTexCoordArray(0, texcoords);
    
    x = x * texScale;
    y = y * texScale;
    z = z * texScale;

    // x y 
    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(x,   0.0));
    texcoords->push_back(osg::Vec2(x,   y));
    texcoords->push_back(osg::Vec2(0.0, y));
    
    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(y,   0.0));
    texcoords->push_back(osg::Vec2(y,   x));
    texcoords->push_back(osg::Vec2(0.0, x));
    
    // x z
    texcoords->push_back(osg::Vec2(0.0, z));
    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(x,   0.0));
    texcoords->push_back(osg::Vec2(x  , z));

    texcoords->push_back(osg::Vec2(0.0, z));
    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(x,   0.0));
    texcoords->push_back(osg::Vec2(x  , z));
   
    // y z
    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(z,   0.0));
    texcoords->push_back(osg::Vec2(z,   y));
    texcoords->push_back(osg::Vec2(0.0, y));

    texcoords->push_back(osg::Vec2(0.0, 0.0));
    texcoords->push_back(osg::Vec2(z,   0.0));
    texcoords->push_back(osg::Vec2(z,   y));
    texcoords->push_back(osg::Vec2(0.0, y));

    // top 
    verts->push_back(osg::Vec3(xMax, yMax, zMax)); 
    normals->push_back(up);
    verts->push_back(osg::Vec3(xMin, yMax, zMax));
    normals->push_back(up);
    verts->push_back(osg::Vec3(xMin, yMin, zMax));
    normals->push_back(up);
    verts->push_back(osg::Vec3(xMax, yMin, zMax));
    normals->push_back(up);

    // bottom 
    verts->push_back(osg::Vec3(xMax, yMax, zMin));
    normals->push_back(down);
    verts->push_back(osg::Vec3(xMin, yMax, zMin));
    normals->push_back(down);
    verts->push_back(osg::Vec3(xMin, yMin, zMin));
    normals->push_back(down);
    verts->push_back(osg::Vec3(xMax, yMin, zMin));
    normals->push_back(down);

    // front 
    verts->push_back(osg::Vec3(xMin, yMin, zMax));
    normals->push_back(front);
    verts->push_back(osg::Vec3(xMin, yMin, zMin));
    normals->push_back(front);
    verts->push_back(osg::Vec3(xMax, yMin, zMin));
    normals->push_back(front);
    verts->push_back(osg::Vec3(xMax, yMin, zMax));
    normals->push_back(front);

    // back 
    verts->push_back(osg::Vec3(xMax, yMax, zMax));
    normals->push_back(back);
    verts->push_back(osg::Vec3(xMax, yMax, zMin));
    normals->push_back(back);
    verts->push_back(osg::Vec3(xMin, yMax, zMin));
    normals->push_back(back);
    verts->push_back(osg::Vec3(xMin, yMax, zMax));
    normals->push_back(back);

    // left 
    verts->push_back(osg::Vec3(xMin, yMax, zMax));
    normals->push_back(left);
    verts->push_back(osg::Vec3(xMin, yMax, zMin));
    normals->push_back(left);
    verts->push_back(osg::Vec3(xMin, yMin, zMin));
    normals->push_back(left);
    verts->push_back(osg::Vec3(xMin, yMin, zMax));
    normals->push_back(left);

    // right 
    verts->push_back(osg::Vec3(xMax, yMax, zMax));
    normals->push_back(right);
    verts->push_back(osg::Vec3(xMax, yMax, zMin));
    normals->push_back(right);
    verts->push_back(osg::Vec3(xMax, yMin, zMin));
    normals->push_back(right);
    verts->push_back(osg::Vec3(xMax, yMin, zMax));
    normals->push_back(right);

    primitive->setCount(4 * 6);
    geometry->dirtyBound();
    
    return geometry;
}

osg::ref_ptr<osg::Geometry> ModelHandler::makeQuad(float width, float height,
        osg::Vec4 color, osg::Vec3 pos)
{
    osg::Geometry * geo = new osg::Geometry();
    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(pos);
    verts->push_back(pos + osg::Vec3(width,0,0));
    verts->push_back(pos + osg::Vec3(width,0,height));
    verts->push_back(pos + osg::Vec3(0,0,height));

    geo->setVertexArray(verts);

    osg::DrawElementsUInt * ele = new osg::DrawElementsUInt(
            osg::PrimitiveSet::QUADS,0);

    ele->push_back(0);
    ele->push_back(1);
    ele->push_back(2);
    ele->push_back(3);
    geo->addPrimitiveSet(ele);

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);

    osg::TemplateIndexArray<unsigned int,osg::Array::UIntArrayType,4,4> *colorIndexArray;
    colorIndexArray = new osg::TemplateIndexArray<unsigned int,
            osg::Array::UIntArrayType,4,4>;
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);
    colorIndexArray->push_back(0);

    geo->setColorArray(colors);
    geo->setColorIndices(colorIndexArray);
    geo->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    osg::Vec2Array* texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0,0));
    texcoords->push_back(osg::Vec2(1,0));
    texcoords->push_back(osg::Vec2(1,1));
    texcoords->push_back(osg::Vec2(0,1));
    geo->setTexCoordArray(0,texcoords);

    return geo;
}

void ModelHandler::openDoor()
{
    if (_activeDoor < 0 || _activeDoor >= (int)_leftdoorPat.size() || _activeDoor >= (int)_rightdoorPat.size())
    {
        return;
    }

    osg::PositionAttitudeTransform *lpat, *rpat;
    lpat = _leftdoorPat[_activeDoor];
    rpat = _rightdoorPat[_activeDoor];

    lpat->setPosition(lpat->getPosition() + lpat->getAttitude() * osg::Vec3(DOOR_SPEED,0,0));
    rpat->setPosition(rpat->getPosition() + rpat->getAttitude() * osg::Vec3(-DOOR_SPEED,0,0));

    _doorDist += DOOR_SPEED;
}

void ModelHandler::closeDoor()
{
    if (_activeDoor < 0 || _activeDoor >= (int)_leftdoorPat.size() || _activeDoor >= (int)_rightdoorPat.size())
    {
        return;
    }
    
    osg::PositionAttitudeTransform *lpat, *rpat;
    lpat = _leftdoorPat[_activeDoor];
    rpat = _rightdoorPat[_activeDoor];
    
    if (_doorDist - DOOR_SPEED < 0)
    {
        _doorDist = 0;
        lpat->setPosition(lpat->getPosition() + lpat->getAttitude() * osg::Vec3(-_doorDist,0,0));
        rpat->setPosition(rpat->getPosition() + rpat->getAttitude() * osg::Vec3( _doorDist,0,0));
    }
    else
    {
        _doorDist -= DOOR_SPEED;
        lpat->setPosition(lpat->getPosition() + lpat->getAttitude() * osg::Vec3(-DOOR_SPEED,0,0));
        rpat->setPosition(rpat->getPosition() + rpat->getAttitude() * osg::Vec3(DOOR_SPEED,0,0));
    }


}

void ModelHandler::setMode(Mode mode)
{
    _alliesSwitch[_activeDoor]->setAllChildrenOff();
    _aliensSwitch[_activeDoor]->setAllChildrenOff();
    _checkersSwitch[_activeDoor]->setAllChildrenOff();
    
    _mode = mode;
    if (mode == ALLY)
    {
        _alliesSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }

    else if (mode == ALIEN)
    {
        _aliensSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }

    else if (mode == CHECKER)
    {
        _checkersSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }

/*    _lightColor = _lightSwitch[_activeDoor]->getNumChildren() - 1;
    _lightSwitch[_activeDoor]->setValue((int)_lightColor, true);
    _leftdoorSwitch[_activeDoor]->setValue((int)_lightColor, true);
*/
}

void ModelHandler::setSwitched(bool switched)
{
    _switched = switched;
}

osg::ref_ptr<osg::Geode> ModelHandler::getActiveObject()
{
    return _activeObject;
}

void ModelHandler::setScore(int score)
{
    char buf[10];
    sprintf(buf, "%d", score);
    std::string text = "Score: ";
    text += buf;
    _scoreText->setText(text);
}

void ModelHandler::flashActiveLight()
{
    if (_lightSwitch[_activeDoor]->getValue(WHITE))
    {
        _lightSwitch[_activeDoor]->setSingleChildOn(GREY);
        _leftdoorSwitch[_activeDoor]->setSingleChildOn(GREY);
    }
    else
    {
        _lightSwitch[_activeDoor]->setSingleChildOn(WHITE);
        _leftdoorSwitch[_activeDoor]->setSingleChildOn(WHITE);
    }
}

void ModelHandler::flashCheckers()
{
    if (_checkersSwitch[_activeDoor]->getValue(0))
    {
        _checkersSwitch[_activeDoor]->setValue(0, false);
        _checkersSwitch[_activeDoor]->setValue(1, true);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(1));
        if (geode)
        {
            _activeObject = geode;
        }
    }
    else
    {
        _checkersSwitch[_activeDoor]->setValue(0, true);
        _checkersSwitch[_activeDoor]->setValue(1, false);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }
}

void ModelHandler::flashAlien()
{
    if (_aliensSwitch[_activeDoor]->getValue(0))
    {
        _aliensSwitch[_activeDoor]->setValue(0, false);
        _aliensSwitch[_activeDoor]->setValue(1, true);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(1));
        if (geode)
        {
            _activeObject = geode;
        }
    }
    else
    {
        _aliensSwitch[_activeDoor]->setValue(0, true);
        _aliensSwitch[_activeDoor]->setValue(1, false);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }
}

void ModelHandler::flashAlly()
{
    if (_alliesSwitch[_activeDoor]->getValue(0))
    {
        _alliesSwitch[_activeDoor]->setValue(0, false);
        _alliesSwitch[_activeDoor]->setValue(1, true);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(1));
        if (geode)
        {
            _activeObject = geode;
        }
    }
    else
    {
        _alliesSwitch[_activeDoor]->setValue(0, true);
        _alliesSwitch[_activeDoor]->setValue(1, false);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }

    }
}

void ModelHandler::setActiveDoor(int doorNum)
{
    _activeDoor = doorNum;

    if (_mode == ALLY)
    {
        _alliesSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }

    else if (_mode == ALIEN)
    {
        _aliensSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }

    else if (_mode == CHECKER)
    {
        _checkersSwitch[_activeDoor]->setSingleChildOn(0);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }


}

void ModelHandler::setAlien(bool val)
{
    _aliensSwitch[_activeDoor]->setValue(0, val);
    _aliensSwitch[_activeDoor]->setValue(1, val);

    if (val) // only show one child
    {
        _aliensSwitch[_activeDoor]->setValue(1, false);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }
}

void ModelHandler::setAlly(bool val)
{
    _alliesSwitch[_activeDoor]->setValue(0, val);
    _alliesSwitch[_activeDoor]->setValue(1, val);

    if (val) // only show one child
    {
        _alliesSwitch[_activeDoor]->setValue(1, false);

        osg::Geode * geode;
        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
        if (geode)
        {
            _activeObject = geode;
        }
    }
}

void ModelHandler::setLight(bool val)
{
    //_lightColor = _lightSwitch[_activeDoor]->getNumChildren() - 1;
    //_lightSwitch[_activeDoor]->setValue((int)_lightColor, true);
    //_leftdoorSwitch[_activeDoor]->setValue((int)_lightColor, true);

    if (val && _switched && _mode == ALIEN)
    {
        Mode mode = ALLY;
        _lightColor = BLUE;
    }
    else if (val && _switched && _mode == ALLY)
    {
        Mode mode = ALIEN;
        _lightColor = RED;
    }
    else
    {
        _lightColor = (int)_mode;
    }

    if (val)
    {
        _lightSwitch[_activeDoor]->setValue(0, false);
        _lightSwitch[_activeDoor]->setAllChildrenOff();
        _lightSwitch[_activeDoor]->setValue(_lightColor, true);

        _leftdoorSwitch[_activeDoor]->setValue(0, false);
        _leftdoorSwitch[_activeDoor]->setAllChildrenOff();
        _leftdoorSwitch[_activeDoor]->setValue(_lightColor, true);
    }
    else
    {
        _lightSwitch[_activeDoor]->setAllChildrenOff();
        _lightSwitch[_activeDoor]->setValue(GREY, true);
        _lightSwitch[_activeDoor]->setValue(_lightColor, false);

        _leftdoorSwitch[_activeDoor]->setAllChildrenOff();
        _leftdoorSwitch[_activeDoor]->setValue(GREY, true);
        _leftdoorSwitch[_activeDoor]->setValue(_lightColor, false);
    }
}

float ModelHandler::getDoorDistance()
{
    return _doorDist;
}

bool ModelHandler::doorInView()
{
    return (_viewedDoor == _activeDoor);//_doorInView;
}

int ModelHandler::getViewedDoor()
{
    return _viewedDoor;
}

void ModelHandler::turnLeft()
{
    _turningLeft = true;
    _viewedDoor = (_viewedDoor + 1) % NUM_DOORS;

    if (_viewedDoor == -1) _viewedDoor += NUM_DOORS;

    
    float angle = 2 * M_PI / NUM_DOORS;
    float offset = (_viewedDoor - 4);
    for (int i = 0; i < NUM_DOORS; ++i)
    {
        if (_audioHandler)
        {
            _audioHandler->update(i + DING_OFFSET, (i - offset) * angle);
            _audioHandler->update(i + EXPLOSION_OFFSET, (i - offset) * angle);
        }
    }
}

void ModelHandler::turnRight()
{
    _turningRight = true;
    _viewedDoor = (_viewedDoor - 1) % NUM_DOORS;
    if (_viewedDoor == -1) _viewedDoor += NUM_DOORS;

    float angle = 2 * M_PI / NUM_DOORS;
    float offset = (_viewedDoor - 4);
    for (int i = 0; i < NUM_DOORS; ++i)
    {
        if (_audioHandler)
        {
            _audioHandler->update(i + DING_OFFSET, (i - offset) * angle);
            _audioHandler->update(i + EXPLOSION_OFFSET, (i - offset) * angle);
        }
    }
}

void ModelHandler::showScore(bool b)
{
    if (b)
    {
        _scoreSwitch->setAllChildrenOn();
    }
    else
    {
        _scoreSwitch->setAllChildrenOff();
    }
}

};

