#include "ElevatorRoom.h"

#define DING_OFFSET 1
#define EXPLOSION_OFFSET 9
#define LASER_OFFSET 17

ElevatorRoom * ElevatorRoom::_myPtr = NULL;

CVRPLUGIN(ElevatorRoom)

using namespace cvr;
using namespace osg;
using namespace std;

ElevatorRoom::ElevatorRoom()
{
    _myPtr = this;
    _loaded = false;
    _audioHandler = NULL;
}

ElevatorRoom::~ElevatorRoom()
{
}

ElevatorRoom * ElevatorRoom::instance()
{
    return _myPtr;
}

bool ElevatorRoom::init()
{
    // Setup menus
    _elevatorMenu = new SubMenu("Elevator Room");

    PluginHelper::addRootMenuItem(_elevatorMenu);

    _loadButton = new MenuButton("Load");
    _loadButton->setCallback(this);
    _elevatorMenu->addItem(_loadButton);

    _clearButton = new MenuButton("Clear");
    _clearButton->setCallback(this);
    _elevatorMenu->addItem(_clearButton);

    _checkerSpeedRV = new MenuRangeValue("Checker flash speed: ", 10, 30, 15);
    _checkerSpeedRV->setCallback(this);
    _elevatorMenu->addItem(_checkerSpeedRV);

    _alienChanceRV = new MenuRangeValue("Chance of alien: ", 0, 100, 50);
    _alienChanceRV->setCallback(this);
    _elevatorMenu->addItem(_alienChanceRV);
    
    _alienChance = 50;
    _allyChance  = 40;
    _checkChance  = 10;

    char str[50];
    sprintf(str, "Alien: %d  Astro: %d  Checker: %d", _alienChance, _allyChance, _checkChance);
    _chancesText = new MenuText(str);
    _elevatorMenu->addItem(_chancesText);
   
    _checkSpeed = 15;
    
    // Load from config
    _dataDir = ConfigManager::getEntry("Plugin.ElevatorRoom.DataDir");
    _dataDir = _dataDir + "/";

    _debug = (ConfigManager::getEntry("Plugin.ElevatorRoom.Debug") == "on");

    _geoRoot = new osg::MatrixTransform();
    
    float xscale, yscale, zscale, xpos, ypos, zpos;

    xscale = ConfigManager::getFloat("x", "Plugin.ElevatorRoom.Scale", 100.0);
    yscale = ConfigManager::getFloat("y", "Plugin.ElevatorRoom.Scale", 100.0);
    zscale = ConfigManager::getFloat("z", "Plugin.ElevatorRoom.Scale", 100.0);

    xpos = ConfigManager::getFloat("x", "Plugin.ElevatorRoom.Position", 0.0);
    ypos = ConfigManager::getFloat("y", "Plugin.ElevatorRoom.Position", 0.0);
    zpos = ConfigManager::getFloat("z", "Plugin.ElevatorRoom.Position", -100.0);

    osg::Vec3 scale = osg::Vec3(xscale, yscale, zscale);

    osg::Matrix mat;
    mat.makeScale(scale);
    _geoRoot->setMatrix(mat);
    mat.makeTranslate(osg::Vec3(xpos, ypos, zpos));
    _geoRoot->postMult(mat);
    
    PluginHelper::getObjectsRoot()->addChild(_geoRoot);

    _activeDoor = -1;
    _isOpening = true;
    _doorDist = 0;
    _pauseStart = PluginHelper::getProgramDuration();
    _pauseLength = -1;
    _mode = NONE;
    _score = 0;
    _hit = false;
    _avatarFlashPerSec = 10;
    _lightFlashPerSec = 7;

    srand(time(NULL));

    // Sound
    
    if (ComController::instance()->isMaster())
    {
        _audioHandler = new AudioHandler();
        _audioHandler->connectServer();
        osg::Vec3 handPos, headPos, headDir, handDir;
        handPos = cvr::PluginHelper::getHandMat().getTrans();
        headPos = cvr::PluginHelper::getHeadMat().getTrans();
        headDir = osg::Vec3(0, 0, -1); 
        handDir = handPos - headPos;

        // user position
        _audioHandler->loadSound(0, headDir, headPos);
        // laser sound
        _audioHandler->loadSound(17, handDir, handPos);
    }

/*    if (ComController::instance()->isMaster())
    {
        std::string server = ConfigManager::getEntry("value", "Plugin.ElevatorRoom.Server", "");
        int port = ConfigManager::getInt("value","Plugin.ElevatorRoom.Port", 0);
    
        if (!oasclient::ClientInterface::isInitialized())
        {
            if (!oasclient::ClientInterface::initialize(server, port))
            {
                std::cerr << "Could not set up connection to sound server!\n" << std::endl;
                _soundEnabled = false;
            }
        }
        else
        {
            std::cerr << "Sound server already initialized!" << std::endl;
        }
        
        oasclient::Listener::getInstance().setGlobalRenderingParameters(
            oasclient::Listener::DEFAULT_REFERENCE_DISTANCE, 1000);

        oasclient::Listener::getInstance().setGlobalRenderingParameters(
            oasclient::Listener::DEFAULT_ROLLOFF, 0.001);

        oasclient::Listener::getInstance().setOrientation(0, 1, 0, 0, 0, 1);

        oasclient::Listener::getInstance().setGlobalRenderingParameters(
            oasclient::Listener::DOPPLER_FACTOR, 0);

        std::string path, file;
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();

        file = ConfigManager::getEntry("file", "Plugin.ElevatorRoom.DingSound", "");
        _ding = new oasclient::Sound(_dataDir, file);
        if (!_ding->isValid())
        {
            std::cerr << "Could not create click sound!\n" << std::endl;
            _soundEnabled = false;
        }
        _ding->setGain(1.5);

        file = ConfigManager::getEntry("file", "Plugin.ElevatorRoom.LaserSound", "");
        _laser= new oasclient::Sound(_dataDir, file);
        if (!_laser->isValid())
        {
            std::cerr << "Could not create click sound!\n" << std::endl;
            _soundEnabled = false;
        }

        file = ConfigManager::getEntry("file", "Plugin.ElevatorRoom.HitSound", "");
        _hitSound = new oasclient::Sound(_dataDir, file);
        if (!_hitSound->isValid())
        {
            std::cerr << "Could not create click sound!\n" << std::endl;
            _soundEnabled = false;
        }


        PluginHelper::getObjectsRoot()->addChild(pat);
        pat->setPosition(osg::Vec3(0,0,0));
        pat->addChild(_ding);
//        pat->setUpdateCallback(new oasclient::SoundUpdateCallback(_ding));
        osg::NodeCallback *cb = new oasclient::SoundUpdateCallback(_ding);
        _ding->setUpdateCallback(cb);
        
    }
    */
    return true;
}

void ElevatorRoom::connectToServer()
{
    // get server address and port number
    string server_addr = ConfigManager::getEntry("Plugin.Maze2.EOGDataServerAddress");
    int port_number = ConfigManager::getInt("Plugin.Maze2.EOGDataServerPort", 8084);
    if( server_addr == "" ) server_addr = "127.0.0.1";
    
    cerr << "Maze2::ECGClient::Server address: " << server_addr << endl;
    cerr << "Maze2::ECGClient::Port number: " << port_number << endl;

    // build up socket communications
    int portno = port_number, protocol = SOCK_STREAM;
    struct sockaddr_in server;
    struct hostent *hp;

    _sockfd = socket(AF_INET, protocol, 0);
    if (_sockfd < 0)
    {
        cerr << "Maze2::ECGClient::connect(): Can't open socket." << endl;
        return;
    }

    server.sin_family = AF_INET;
    hp = gethostbyname(server_addr.c_str());
    if (hp == 0)
    {
        cerr << "Maze2::ECGClient::connect(): Unknown host." << endl;
        close(_sockfd);
        return;
    }
    memcpy((char *)&server.sin_addr, (char *)hp->h_addr, hp->h_length);
    server.sin_port = htons((unsigned short)portno);

    // connect to ECG data server
    if (connect(_sockfd, (struct sockaddr *) &server, sizeof (server)) < 0)
    {
	cerr << "Maze2::ECGClient::connect(): Failed connect to server" << endl;
        close(_sockfd);
        return;
    }

    cerr << "Maze2::ECGClient::Successfully connected to EOG Data Server." << endl;
    _connected = true;
}

void ElevatorRoom::loadModels()
{
    std::string _wallTex, _floorTex, _ceilingTex, _doorTex,
            _alienTex, _allyTex, _checkTex1, _checkTex2, _elevTex;

    _wallTex = ConfigManager::getEntry("Plugin.ElevatorRoom.WallTexture");
    _floorTex = ConfigManager::getEntry("Plugin.ElevatorRoom.FloorTexture");
    _ceilingTex = ConfigManager::getEntry("Plugin.ElevatorRoom.CeilingTexture");
    _doorTex = ConfigManager::getEntry("Plugin.ElevatorRoom.DoorTexture");
    _elevTex = ConfigManager::getEntry("Plugin.ElevatorRoom.ElevatorTexture");

    _alienTex = ConfigManager::getEntry("Plugin.ElevatorRoom.AlienTexture");
    _allyTex = ConfigManager::getEntry("Plugin.ElevatorRoom.AllyTexture");

    _checkTex1 = ConfigManager::getEntry("Plugin.ElevatorRoom.CheckerTexture1");
    _checkTex2 = ConfigManager::getEntry("Plugin.ElevatorRoom.CheckerTexture2");


    osg::Vec4 grey     = osg::Vec4(0.7, 0.7, 0.7, 1.0),
              brown    = osg::Vec4(0.3, 0.15, 0.0, 1.0),
              darkgrey = osg::Vec4(0.3, 0.3, 0.3, 1.0),
              red      = osg::Vec4(1,0,0,1), 
              blue     = osg::Vec4(0,0,1,1),
              green    = osg::Vec4(0,1,0,1),
              yellow   = osg::Vec4(1,1,0,1);

    float roomRad = 6.0, angle = 2 * M_PI / NUM_DOORS;

    osg::ref_ptr<osg::PositionAttitudeTransform> pat;
    osg::ref_ptr<osg::ShapeDrawable> drawable;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geo;

    osg::ref_ptr<osg::Texture2D> tex;
    osg::ref_ptr<osg::Image> img;
    osg::ref_ptr<osg::StateSet> state;


    // Lights
    osg::Sphere * shape = new osg::Sphere(osg::Vec3(0,-4.75, 4.0), 0.2);
    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        drawable = new osg::ShapeDrawable(shape);
        drawable->setColor(osg::Vec4(0.5, 0.5, 0.5, 1.0));
        geode = new osg::Geode();
        geode->addDrawable(drawable);

        osg::ref_ptr<osg::ShapeDrawable> redDrawable, greenDrawable, yellowDrawable;
        osg::ref_ptr<osg::Geode> redGeode, greenGeode, yellowGeode;

        redDrawable = new osg::ShapeDrawable(shape);
        redDrawable->setColor(red);
        redGeode = new osg::Geode();
        redGeode->addDrawable(redDrawable);

        greenDrawable = new osg::ShapeDrawable(shape);
        greenDrawable->setColor(green);
        greenGeode = new osg::Geode();
        greenGeode->addDrawable(greenDrawable);

        yellowDrawable = new osg::ShapeDrawable(shape);
        yellowDrawable->setColor(yellow);
        yellowGeode = new osg::Geode();
        yellowGeode->addDrawable(yellowDrawable);

        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0.0, 0.0, 1.0)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));

        osg::ref_ptr<osg::Switch> switchNode;
        switchNode = new osg::Switch();
        
        switchNode->addChild(geode, true);
        switchNode->addChild(redGeode, false);
        switchNode->addChild(greenGeode, false);       
        switchNode->addChild(yellowGeode, false);       

        pat->addChild(switchNode);
        _geoRoot->addChild(pat);

        _lightSwitch.push_back(switchNode);

        _lights.push_back(drawable);
        
        
        // Sound
        
        if (_audioHandler)
        {
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
            _audioHandler->loadSound(i + DING_OFFSET, dir, pos);
        }
    }


    // Aliens 
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, grey);
    geode->addDrawable(geo);

    osg::ref_ptr<osg::Geode> redGeode = new Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, red);
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
        
        if (_audioHandler)
        {
            osg::Vec3 pos = osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0);
            osg::Vec3 dir = pos - osg::Vec3(0,0,0);

            // 9 - 16 explosion sounds
            _audioHandler->loadSound(i + EXPLOSION_OFFSET, dir, pos);
        }
    }   


    // Allies 
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, grey);
    geode->addDrawable(geo);
    
    redGeode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, red);
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


    // Checkerboards 
    geode = new osg::Geode();
    geo = drawBox(osg::Vec3(0, -6, 0.5), 1.0, 0.01, 1.0, grey);
    geode->addDrawable(geo);

    osg::Geode * geode2 = new osg::Geode();
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

    // Walls
    
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
    geo = drawBox(osg::Vec3(3.0, -5.0, 1.0), 3.25, 0.5, 4.0, grey, wallTexScale); 
    geode->addDrawable(geo);
    
    // Right front
    geo = drawBox(osg::Vec3(-3.0, -5.0, 1.0), 3.25, 0.5, 4.0, grey, wallTexScale);
    geode->addDrawable(geo);

    // Top
    geo = drawBox(osg::Vec3(0.0, -5.0, 4.5), 9.0, 0.5, 3.0, grey, wallTexScale);
    geode->addDrawable(geo);

    
    // Elevator
    
    osg::Geode * elevatorGeode = new osg::Geode();
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
    geo = drawBox(osg::Vec3( 1.25, -7.0, 1.5), 0.5, 4.0, 5.0, grey);
    elevatorGeode->addDrawable(geo);

    // Right side 
    geo = drawBox(osg::Vec3(-1.25, -7.0, 1.5), 0.5, 4.0, 5.0, grey);
    elevatorGeode->addDrawable(geo);

    // Back
    geo = drawBox(osg::Vec3(0.0, -9.25, 1.5), 3.0, 0.5, 5.0, grey);
    elevatorGeode->addDrawable(geo);

    // Elevator floor
    geo = drawBox(osg::Vec3(0.0, -7.0, -1.15), 3.0, 3.75, 0.5, grey);
    elevatorGeode->addDrawable(geo);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->addChild(geode);
        pat->addChild(elevatorGeode);
        _geoRoot->addChild(pat);
    }


    // Ceiling 
    
    geode = new osg::Geode();

    geo = drawBox(osg::Vec3(0.0, 0.0, 5.0), 40.0, 40.0, 0.1, grey);
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
    

    // Floor

    geode = new osg::Geode();

    geo = drawBox(osg::Vec3(0.0, 0.0, -1.0), 40.0, 40.0, 0.1, grey);
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


    // Doors
    
    osg::ref_ptr<osg::Geometry> ldoorGeo, rdoorGeo;

    ldoorGeo = drawBox(osg::Vec3( 0.75, -5.2, 1.0), 1.5, 0.5, 4.0, grey);
    rdoorGeo = drawBox(osg::Vec3(-0.75, -5.2, 1.0), 1.5, 0.5, 4.0, grey);

    for (int i = 0; i < NUM_DOORS; ++i)
    {
        pat = new osg::PositionAttitudeTransform();
        geode = new osg::Geode();

        tex = new osg::Texture2D();
        img = osgDB::readImageFile(_dataDir + _doorTex);
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

        geode->addDrawable(rdoorGeo);
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->addChild(geode);
        _rightdoorPat.push_back(pat);
        _geoRoot->addChild(pat);


        pat = new osg::PositionAttitudeTransform();
        geode = new osg::Geode();

        state = geode->getOrCreateStateSet();
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
        state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

        geode->addDrawable(ldoorGeo);
        pat->setAttitude(osg::Quat(i * angle, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Quat(i * angle, osg::Vec3(0, 0, 1)) * osg::Vec3(0.0, -roomRad, 0.0));
        pat->addChild(geode);
        _leftdoorPat.push_back(pat);
        _geoRoot->addChild(pat);
    }


    // Score text
    
    osg::Vec3 pos = osg::Vec3(-500, 0, 300);
    _scoreText = new osgText::Text();
    _scoreText->setText("Score: 0");
    _scoreText->setCharacterSize(20);
    _scoreText->setAlignment(osgText::Text::LEFT_CENTER);
    _scoreText->setPosition(pos);
    _scoreText->setColor(osg::Vec4(1,1,1,1));
    _scoreText->setBackdropColor(osg::Vec4(0,0,0,0));
    _scoreText->setAxisAlignment(osgText::Text::XZ_PLANE);

    float width = 200, height = 50;
    osg::ref_ptr<osg::Geometry> quad = makeQuad(width, height, 
        osg::Vec4(1.0,1.0,1.0,0.5), pos - osg::Vec3(10, 0, 25));

    pat = new osg::PositionAttitudeTransform();
    geode = new osg::Geode();

    geode->addDrawable(_scoreText);
    geode->addDrawable(quad);

    geode->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    pat->addChild(geode);
    PluginHelper::getScene()->addChild(pat);


    // Crosshair
    
    osg::ref_ptr<osg::Vec3Array> _verts;
    osg::ref_ptr<osg::Vec4Array> _colors;
    osg::ref_ptr<osg::Vec3Array> _normals;
    osg::ref_ptr<osg::DrawArrays> _primitive;
    osg::ref_ptr<osg::Geometry> _geometry;
    osg::Geode * chGeode = new osg::Geode();

    _verts = new osg::Vec3Array(0);
    _colors = new osg::Vec4Array(1);
    _normals = new osg::Vec3Array(0);
    _primitive = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 0);
    _geometry = new osg::Geometry();

    _geometry->setVertexArray(_verts.get());
    _geometry->setColorArray(_colors.get());
    _geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    _geometry->setNormalArray(_normals.get());
    _geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    _geometry->setUseDisplayList(false);
    _geometry->addPrimitiveSet(_primitive.get());

    _verts->clear();
    _normals->clear();
    
    float y = 1;
    width = 200;
    height = 50;
    osg::Vec3 normal = osg::Vec3(0, -1, 0);

    quad = makeQuad(width, height, 
        osg::Vec4(1.0,1.0,1.0,0.5), osg::Vec3(0,0,0) - osg::Vec3(10, 0, 25));
    
    chGeode->addDrawable(quad);

    PluginHelper::getScene()->addChild(pat);

    _loaded = true;
}

void ElevatorRoom::clear()
{
}

void ElevatorRoom::menuCallback(MenuItem * item)
{
    if(item == _loadButton)
    {
        if (!_loaded)
        {
            loadModels();
            _loaded = true;
        }
    }

    else if (item == _clearButton)
    {
        if (_loaded)
        {
            clear();
            _loaded = false;
        }
    }

    else if(item == _checkerSpeedRV)
    {
        _checkSpeed = (int)_checkerSpeedRV->getValue();
    }

    else if(item == _alienChanceRV)
    {
        int newVal = _alienChanceRV->getValue();
        if (_alienChance + _allyChance + _checkChance <= 100 &&
            newVal > -1  && 100 - newVal - _checkChance > -1  && _checkChance > -1)
        {
            _alienChance = newVal;
            _allyChance = 100 - _alienChance - _checkChance;

            char str[50];
            sprintf(str, "Alien: %d  Astro: %d  Checker: %d", _alienChance, _allyChance, _checkChance);
            _chancesText->setText(str);
        }

    }

}

void ElevatorRoom::preFrame()
{
    if (!_loaded)
        return;
    
    // Pick a door to open 
    if (_activeDoor < 0)
    {
        if ((PluginHelper::getProgramDuration() - _pauseStart) > _pauseLength)
        {
            if (_audioHandler)
            {
                _audioHandler->playSound(_activeDoor + DING_OFFSET, "ding");
            }

            _activeDoor = rand() % NUM_DOORS;

            int num = rand() % 10;

            if (num <= 3)
            {
                if (_debug)
                {
                    std::cout << _activeDoor << " - alien" << std::endl;
                }

                _aliensSwitch[_activeDoor]->setSingleChildOn(0);

                _alliesSwitch[_activeDoor]->setAllChildrenOff();
                _checkersSwitch[_activeDoor]->setAllChildrenOff();

                _mode = ALIEN;

                osg::ref_ptr<osg::Geode > geode;
                geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
                if (geode)
                {
                    _activeObject = geode;
                }
            }
            else if (num <= 7)
            {
                if (_debug)
                {
                    std::cout << _activeDoor << " - ally" << std::endl;
                }

                _alliesSwitch[_activeDoor]->setSingleChildOn(0);

                _aliensSwitch[_activeDoor]->setAllChildrenOff();
                _checkersSwitch[_activeDoor]->setAllChildrenOff();

                _mode = ALLY;

                osg::ref_ptr<osg::Geode > geode;
                geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
                if (geode)
                {
                    _activeObject = geode;
                }
            }
            else
            {
                if (_debug)
                {
                    std::cout << _activeDoor << " - checker " << std::endl;
                }

                _checkersSwitch[_activeDoor]->setSingleChildOn(0);

                _alliesSwitch[_activeDoor]->setAllChildrenOff();
                _aliensSwitch[_activeDoor]->setAllChildrenOff();

                _mode = CHECKER;

                osg::ref_ptr<osg::Geode > geode;
                geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(0));
                if (geode)
                {
                    _activeObject = geode;
                }
            }
            _flashCount = 0;
        }

        if (_activeDoor <= _lights.size())
        {
            _lightSwitch[_activeDoor]->setValue((int)_mode, true);
            _flashStartTime = PluginHelper::getProgramDuration();
            _pauseStart = PluginHelper::getProgramDuration();
            _pauseLength = LIGHT_PAUSE_LENGTH;
        }

    }
    
    // Handle light flashes
    if (_activeDoor >= 0 && _activeDoor < NUM_DOORS &&
        (PluginHelper::getProgramDuration() - _pauseStart) < _pauseLength)
    {
        if (PluginHelper::getProgramDuration() - _flashStartTime > 1 / _lightFlashPerSec)
        {
            if (_lightSwitch[_activeDoor]->getValue(_mode))
            {
                _lightSwitch[_activeDoor]->setSingleChildOn(NONE);
                _flashStartTime = PluginHelper::getProgramDuration();
            }
            else
            {
                _lightSwitch[_activeDoor]->setSingleChildOn(_mode);
                _flashStartTime = PluginHelper::getProgramDuration();
            }
        }
    }
   
    // Handle door movement and animation
    if (_activeDoor >= 0 && _activeDoor < NUM_DOORS &&
        (PluginHelper::getProgramDuration() - _pauseStart) > _pauseLength)
    {
        _lightSwitch[_activeDoor]->setValue((int)_mode, true);
        
        // Flashing avatars
       
        if (_mode == CHECKER)
        {
            if (PluginHelper::getProgramDuration() - _flashStartTime > 1 / _checkSpeed)
            {
                if (_checkersSwitch[_activeDoor]->getValue(0))
                {
                    _checkersSwitch[_activeDoor]->setValue(0, false);
                    _checkersSwitch[_activeDoor]->setValue(1, true);

                    osg::ref_ptr<osg::Geode > geode;
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

                    osg::ref_ptr<osg::Geode > geode;
                    geode = dynamic_cast<osg::Geode *>(_checkersSwitch[_activeDoor]->getChild(0));
                    if (geode)
                    {
                        _activeObject = geode;
                    }
                }
                _flashCount++;
                _flashStartTime = PluginHelper::getProgramDuration();
            }
        }

        else if (_mode == ALIEN)
        {
            if (_hit)
            {
                if (_flashCount > NUM_ALIEN_FLASH)
                {
                        _aliensSwitch[_activeDoor]->setValue(0, false);
                        _aliensSwitch[_activeDoor]->setValue(1, false);
                }

                else if (PluginHelper::getProgramDuration() - _flashStartTime > 1 / _avatarFlashPerSec)
                {
                    if (_aliensSwitch[_activeDoor]->getValue(0))
                    {
                        _aliensSwitch[_activeDoor]->setValue(0, false);
                        _aliensSwitch[_activeDoor]->setValue(1, true);

                        osg::ref_ptr<osg::Geode > geode;
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

                        osg::ref_ptr<osg::Geode > geode;
                        geode = dynamic_cast<osg::Geode *>(_aliensSwitch[_activeDoor]->getChild(0));
                        if (geode)
                        {
                            _activeObject = geode;
                        }
                    }
                    _flashCount++; 
                    _flashStartTime = PluginHelper::getProgramDuration();
                }
            }
        }

        else if (_mode == ALLY)
        {
            if (_hit)
            {
                if (_flashCount > NUM_ALLY_FLASH)
                {
                        _alliesSwitch[_activeDoor]->setValue(0, true);
                        _alliesSwitch[_activeDoor]->setValue(1, false);
                }

                else if (PluginHelper::getProgramDuration() - _flashStartTime > 1 / _avatarFlashPerSec)
                {
                    if (_alliesSwitch[_activeDoor]->getValue(0))
                    {
                        _alliesSwitch[_activeDoor]->setValue(0, false);
                        _alliesSwitch[_activeDoor]->setValue(1, true);

                        osg::ref_ptr<osg::Geode > geode;
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

                        osg::ref_ptr<osg::Geode > geode;
                        geode = dynamic_cast<osg::Geode *>(_alliesSwitch[_activeDoor]->getChild(0));
                        if (geode)
                        {
                            _activeObject = geode;
                        }
                    }
                    _flashCount++; 
                    _flashStartTime = PluginHelper::getProgramDuration();
                }
            }
        }

        if (_isOpening)
        {
            openDoor(_activeDoor);
            if (_doorDist > 0.8)
            {
                _isOpening = false;
            }
        }
        else
        {
            closeDoor(_activeDoor);
            if (_doorDist < DOOR_SPEED)
            {
                if (_activeDoor <= _lightSwitch.size())
                {
                    _lightSwitch[_activeDoor]->setValue(NONE, true);
                }
                
                _isOpening = true;
                _activeDoor = -1;
                _doorDist = 0;
                _pauseStart = PluginHelper::getProgramDuration();
                _pauseLength = 1 + rand() % 5;
                _hit = false;

                if (_debug)
                {
                    std::cout << "Pause for " << _pauseLength << " seconds" << std::endl;
                }
            }
        }
    }
    
    // Update sound
    if (_audioHandler)
    {
        osg::Vec3 handPos, headPos, headDir, handDir;
        handPos = cvr::PluginHelper::getHandMat().getTrans();
        headPos = cvr::PluginHelper::getHeadMat().getTrans();
        headDir = osg::Vec3(0, 0, -1); 
        handDir = handPos - headPos;

        // user position
        _audioHandler->update(0, headDir, headPos);
        // laser sound
        _audioHandler->update(17, handDir, handPos);
    }
}

void ElevatorRoom::openDoor(int doorNum)
{
    if (doorNum < 0 || doorNum >= (int)_leftdoorPat.size() || doorNum >= (int)_rightdoorPat.size())
    {
        return;
    }

    osg::PositionAttitudeTransform *lpat, *rpat;
    lpat = _leftdoorPat[doorNum];
    rpat = _rightdoorPat[doorNum];

    lpat->setPosition(lpat->getPosition() + lpat->getAttitude() * osg::Vec3(DOOR_SPEED,0,0));
    rpat->setPosition(rpat->getPosition() + rpat->getAttitude() * osg::Vec3(-DOOR_SPEED,0,0));

    _doorDist += DOOR_SPEED;
}

void ElevatorRoom::closeDoor(int doorNum)
{
    if (doorNum < 0 || doorNum >= (int)_leftdoorPat.size() || doorNum >= (int)_rightdoorPat.size())
    {
        return;
    }
    
    osg::PositionAttitudeTransform *lpat, *rpat;
    lpat = _leftdoorPat[doorNum];
    rpat = _rightdoorPat[doorNum];

    lpat->setPosition(lpat->getPosition() + lpat->getAttitude() * osg::Vec3(-DOOR_SPEED,0,0));
    rpat->setPosition(rpat->getPosition() + rpat->getAttitude() * osg::Vec3(DOOR_SPEED,0,0));

    _doorDist -= DOOR_SPEED;
}

bool ElevatorRoom::processEvent(InteractionEvent * event)
{
    if (!_loaded)
        return false;

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();

    if(!tie)
    {
        return true;
    }

    if(tie->getHand() == 0 && tie->getButton() == 0)
    {
        if (tie->getInteraction() == BUTTON_DOWN)
        {
            osg::Vec3 pointerStart, pointerEnd;
            std::vector<IsectInfo> isecvec;
            
            osg::Matrix pointerMat = tie->getTransform();
            pointerStart = pointerMat.getTrans();
            pointerEnd.set(0.0f, 10000.0f, 0.0f);
            pointerEnd = pointerEnd * pointerMat;

            isecvec = getObjectIntersection(cvr::PluginHelper::getScene(),
                    pointerStart, pointerEnd);

            _eventRot = tie->getTransform().getRotate();
            _eventPos = tie->getTransform().getTrans();

            if (isecvec.size() == 0)
            {
                return true;
            }
            else
            {
                if (_activeDoor >= 0)
                {
                    if (isecvec[0].geode == _activeObject && _doorDist > 0)
                    {
                        /*if (_laser)
                        {
                            _laser->play();
                        }*/
                        if (_audioHandler)
                        {
                            _audioHandler->playSound(LASER_OFFSET, "laser");
                        }

                        if (_mode == ALIEN && !_hit)
                        {
                            /*if (_hitSound)
                            {
                                _hitSound->play();
                            }*/
                            if (_audioHandler)
                            {
                                _audioHandler->playSound(_activeDoor + EXPLOSION_OFFSET, "explosion");
                            }

                            std::cout << "Hit!" << std::endl; 
                            _score++;

                            char buf[10];
                            sprintf(buf, "%d", _score);
                            std::string text = "Score: ";
                            text += buf;
                            _scoreText->setText(text);

                            std::cout << "Score: " << _score << std::endl;
                            _hit = true;
                        }
                        else if (_mode == ALLY && !_hit)
                        {
                            /*if (_hitSound)
                            {
                                _hitSound->play();
                            }*/
                            if (_audioHandler)
                            {
                                _audioHandler->playSound(_activeDoor + EXPLOSION_OFFSET, "explosion");
                            }

                            std::cout << "Whoops!" << std::endl;
                            if (_score > 0)
                            {
                                _score--;
                            }

                            char buf[10];
                            sprintf(buf, "%d", _score);
                            std::string text = "Score: ";
                            text += buf;
                            _scoreText->setText(text);

                            std::cout << "Score: " << _score << std::endl;
                            _hit = true;
                        }
                        return true;
                    }
                }
            }
        }
        else if (tie->getInteraction() == BUTTON_DRAG)
        {
            osg::Matrix mat = tie->getTransform();

            osg::Vec3 pos, offset;
            SceneManager::instance()->getPointOnTiledWall(mat,pos);
            offset.y() = -(pos.z() - _eventPos.z()) * 50.0 / SceneManager::instance()->getTiledWallHeight();
            offset = offset * Navigation::instance()->getScale();
            offset = osg::Vec3(0,0,0);
            osg::Matrix m;

            osg::Matrix r;
            r.makeRotate(_eventRot);
            osg::Vec3 pointInit = osg::Vec3(0,1,0);
            pointInit = pointInit * r;
            pointInit.z() = 0.0;

            r.makeRotate(mat.getRotate());
            osg::Vec3 pointFinal = osg::Vec3(0,1,0);
            pointFinal = pointFinal * r;
            pointFinal.z() = 0.0;

            osg::Matrix turn;
            
            if(pointInit.length2() > 0 && pointFinal.length2() > 0)
            {
                pointInit.normalize();
                pointFinal.normalize();
                float dot = pointInit * pointFinal;
                float angle = acos(dot) / 15.0;

                if(dot > 1.0 || dot < -1.0)
                {
                    angle = 0.0;
                }
                else if((pointInit ^ pointFinal).z() < 0)
                {
                    angle = -angle;
                }
                turn.makeRotate(-angle, osg::Vec3(0, 0, 1));
            }

            osg::Matrix objmat =
                    SceneManager::instance()->getObjectTransform()->getMatrix();
            
            osg::Vec3 origin = mat.getTrans();
            origin = _geoRoot->getMatrix().getTrans();

            m.makeTranslate(origin + offset);
            m = objmat * osg::Matrix::translate(-(origin+offset)) * turn * m;
            SceneManager::instance()->setObjectMatrix(m);

            return true;
        }
        else if (tie->getInteraction() == BUTTON_UP)
        {
            return true;
        }
        return true;
    }
    return true;
}

osg::ref_ptr<osg::Geometry> ElevatorRoom::drawBox(osg::Vec3 center, float x, 
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

osg::ref_ptr<osg::Geometry> ElevatorRoom::makeQuad(float width, float height,
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

