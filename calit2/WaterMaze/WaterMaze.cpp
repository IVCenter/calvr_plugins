#include "WaterMaze.h"
#include <string>

using namespace cvr;
using namespace osg;
using namespace std;
using namespace sc;

namespace WaterMaze
{

WaterMaze * WaterMaze::_myPtr = NULL;

CVRPLUGIN(WaterMaze)

WaterMaze::WaterMaze()
{
    _myPtr = this;
    _geoRoot = new osg::MatrixTransform();	

    string name = "aserver";
    //string synthdir = "/Users/demo/workspace/git/collider/synthdefs/mac"; //Octo
    string synthdir = "/Users/administrator/git/collider/synthdefs/mac";

   //_aserver = new SCServer(name, "132.239.235.169" , "57110", synthdir); //Octo
   _aserver = new SCServer(name, "127.0.0.1" , "57110", synthdir);
   _regTileBuf = new Buffer(_aserver, _aserver->nextBufferNum());
   _hiddenTileBuf = new Buffer(_aserver, _aserver->nextBufferNum());

   _aserver->dumpOSC(1);

   _curTile=0;



    //_sppConnected = false;
    _hiddenTile = -1;

    

    float heightOffset = ConfigManager::getFloat("value", 
        "Plugin.WaterMaze.StartingHeight", 300.0);

    osg::Matrixd mat;
    mat.makeTranslate(0, -3000, -heightOffset);
    _geoRoot->setMatrix(mat);
    PluginHelper::getObjectsRoot()->addChild(_geoRoot);
}

WaterMaze::~WaterMaze()
{

    if(_regTileBuf)
	delete _regTileBuf;
    if(_hiddenTileBuf)
	delete _hiddenTileBuf;
    if(_aserver)
	delete _aserver;
}

WaterMaze * WaterMaze::instance()
{
    return _myPtr;
}

bool WaterMaze::init()
{
    
    //_regTileBuf->allocRead("/Users/demo/workspace/svn/libSCresources/projects/watermaze/step.aiff");
    _regTileBuf->allocRead("/Users/administrator/libSCresources/projects/watermaze/step.aiff");

   // _hiddenTileBuf->allocRead("/Users/demo/workspace/svn/libSCresources/projects/watermaze/groove.aiff");
      _hiddenTileBuf->allocRead("/Users/administrator/libSCresources/projects/watermaze/groove.aiff");
    // Setup menus
    _WaterMazeMenu = new SubMenu("Water Maze");

    PluginHelper::addRootMenuItem(_WaterMazeMenu);

    _loadButton = new MenuButton("Load");
    _loadButton->setCallback(this);
    _WaterMazeMenu->addItem(_loadButton);

    _clearButton = new MenuButton("Clear");
    _clearButton->setCallback(this);
    _WaterMazeMenu->addItem(_clearButton);

    _newTileButton = new MenuButton("New Tile");
    _newTileButton->setCallback(this);
    _WaterMazeMenu->addItem(_newTileButton);

    _gridCB = new MenuCheckbox("Show Grid", false);
    _gridCB->setCallback(this);
    _WaterMazeMenu->addItem(_gridCB);

    _resetButton = new MenuButton("Reset");
    _resetButton->setCallback(this);
    _WaterMazeMenu->addItem(_resetButton);

    // extra output messages
    _debug = (ConfigManager::getEntry("Plugin.WaterMaze.Debug") == "on");

    // Sync random number generator
    if(ComController::instance()->isMaster())
    {
        int seed = time(NULL);
		ComController::instance()->sendSlaves(&seed, sizeof(seed));
        srand(seed);
    } 
    else 
    {
        int seed = 0;
		ComController::instance()->readMaster(&seed, sizeof(seed));
        srand(seed);
    }

    // EEG device communication
/*    if (ComController::instance()->isMaster())
    {
        int port = 12345;
        init_SPP(port);
    }
*/
    // Set up models
    widthTile = ConfigManager::getFloat("value", "Plugin.WaterMaze.WidthTile", 2000.0);
    heightTile = ConfigManager::getFloat("value", "Plugin.WaterMaze.HeightTile", 2000.0);
    numWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumWidth", 10.0);
    numHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumHeight", 10.0);
    depth = ConfigManager::getFloat("value", "Plugin.WaterMaze.Depth", 10.0);
    wallHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.WallHeight", 2500.0);
    gridWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.GridWidth", 5.0);

    // Tiles
    osg::Box * box = new osg::Box(osg::Vec3(0,0,0), widthTile, heightTile, depth);
    for (int i = 0; i < numWidth; ++i)
    {
        for (int j = 0; j < numHeight; ++j)
        {
            osg::PositionAttitudeTransform * tilePat = new osg::PositionAttitudeTransform();
            tilePat->setPosition(osg::Vec3((widthTile*i) - (widthTile/2), 
                                           (heightTile*j) - (heightTile/2),
                                            0));

            osg::Switch * boxSwitch = new osg::Switch();
            osg::ShapeDrawable * sd = new osg::ShapeDrawable(box);
            sd->setColor(osg::Vec4(1, 1, 1, 1));
            osg::Geode * geode = new osg::Geode();
            geode->addDrawable(sd);
            boxSwitch->addChild(geode);

            sd = new osg::ShapeDrawable(box);
            sd->setColor(osg::Vec4(0, 1, 0, 1));
            geode = new osg::Geode();
            geode->addDrawable(sd);
            boxSwitch->addChild(geode);

            sd = new osg::ShapeDrawable(box);
            sd->setColor(osg::Vec4(1, 0, 0, 1));
            geode = new osg::Geode();
            geode->addDrawable(sd);
            boxSwitch->addChild(geode);

            tilePat->addChild(boxSwitch);
            _geoRoot->addChild(tilePat);

            osg::Vec3 center;
            center = tilePat->getPosition();
            _tileSwitches[center] = boxSwitch;
        }
    }
    
    // Grid
    _gridSwitch = new osg::Switch();
    for (int i = -1; i < numWidth; ++i)
    {
        box = new osg::Box(osg::Vec3(i * widthTile, heightTile * (numHeight-2) * .5, 0), 
            gridWidth, heightTile * numHeight, depth + 1);
        osg::ShapeDrawable * sd = new osg::ShapeDrawable(box);
        sd->setColor(osg::Vec4(0,0,0,1));
        osg::Geode * geode = new osg::Geode();
        geode->addDrawable(sd);
        _gridSwitch->addChild(geode);
    }

    for (int i = -1; i < numHeight; ++i)
    {
        box = new osg::Box(osg::Vec3(widthTile * (numWidth-2) * .5, i * heightTile, 0), 
            widthTile * numWidth, gridWidth, depth + 1);
        osg::ShapeDrawable * sd = new osg::ShapeDrawable(box);
        sd->setColor(osg::Vec4(0,0,0,1));
        osg::Geode * geode = new osg::Geode();
        geode->addDrawable(sd);
        _gridSwitch->addChild(geode);
    }
    _gridSwitch->setAllChildrenOff();
    _geoRoot->addChild(_gridSwitch); 


    // Walls
    osg::ShapeDrawable * sd;
    osg::Geode * geode;
    osg::Vec3 pos;
    
    // far horizontal
    pos = osg::Vec3(widthTile * (numWidth-2) * 0.5, 
                   (numHeight-1) * heightTile , 
                    wallHeight / 2);
    box = new osg::Box(pos, widthTile * numWidth, 4, wallHeight);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 0.8, 0.8, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _geoRoot->addChild(geode);
     
    // near horizontal
    pos = osg::Vec3(widthTile * (numWidth-2) * 0.5, 
                    (-1) * heightTile, 
                    wallHeight / 2);
    box = new osg::Box(pos, widthTile * numWidth, 4, wallHeight);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 0.8, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _geoRoot->addChild(geode);

    // left vertical
    pos = osg::Vec3((numWidth-1) * widthTile, 
                    heightTile * (numHeight-2) * .5, 
                    wallHeight/2);
    box = new osg::Box(pos, 4, heightTile * numHeight, wallHeight);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(0.8, 1.0, 0.8, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _geoRoot->addChild(geode);

    // right vertical
    pos = osg::Vec3((-1) * widthTile, 
                    heightTile * (numHeight-2) * .5, 
                    wallHeight/2);
    box = new osg::Box(pos, 4, heightTile * numHeight, wallHeight);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(0.8, 0.8, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    _geoRoot->addChild(geode);

    // ceiling
    pos = osg::Vec3((numWidth-2) * widthTile * .5, 
                    (numHeight-2) * heightTile * .5, 
                    wallHeight);
    box = new osg::Box(pos, numWidth * widthTile, numHeight * heightTile, 5);
    sd = new osg::ShapeDrawable(box);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _geoRoot->addChild(geode);
    
    // floor plane
    pos = osg::Vec3((numWidth-2) * widthTile * .5, 
                    (numHeight-2) * heightTile * .5, 
                    0);
    box = new osg::Box(pos, 200000, 200000, 5);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1, .8, .8, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _geoRoot->addChild(geode);

    // sky box
    pos = osg::Vec3((numWidth-2) * widthTile * .5, 
                    (numHeight-2) * heightTile * .5, 
                    0);
    box = new osg::Box(pos, 200000, 200000, 200000);
    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(.8, .8, 1, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    _geoRoot->addChild(geode);
 
    _aserver->dumpOSC(0);

    return true;
}

void WaterMaze::preFrame()
{
   // std::map<string, float> regTileArgs;
   // std::map<string, float> hiddenTileArgs;

    _regTileArgs["bufnum"] = _regTileBuf->getBufNum();
    _hiddenTileArgs["bufnum"] = _hiddenTileBuf->getBufNum();

    

    if (_hiddenTile < 0)
    {
        _hiddenTile = rand() % (int)(numWidth * numHeight);
        std::cout << "Hidden tile = " << _hiddenTile << std::endl;
    }

    osg::Vec3 pos = osg::Vec3(0,0,0) * cvr::PluginHelper::getHeadMat() * 
        PluginHelper::getWorldToObjectTransform() * _geoRoot->getInverseMatrix();

    osg::Vec3 bottomLeft, topRight;
    bottomLeft = osg::Vec3(0,0,0) * _geoRoot->getMatrix();
    topRight = osg::Vec3(widthTile * numWidth, heightTile * numHeight, 0) * 
        _geoRoot->getMatrix();

    float xmin, xmax, ymin, ymax;
    xmin = bottomLeft[0];
    ymin = bottomLeft[1];
    xmax = topRight[0];
    ymax = topRight[1];
    
    int i = 0;
    std::map<osg::Vec3, osg::Switch *>::iterator it;
    for (it = _tileSwitches.begin(); it != _tileSwitches.end(); ++it)
    {
        osg::Vec3 center = it->first;
        xmin = center[0] - widthTile/2;
        xmax = center[0] + widthTile/2;
        ymin = center[1] - heightTile/2;
        ymax = center[1] + heightTile/2;

        
        // Occupied tile
        if (pos[0] > xmin && pos[0] < xmax &&
            pos[1] > ymin && pos[1] < ymax)
        {
            //std::cout << "Standing in tile " << i << std::endl;  
            it->second->setSingleChildOn(2);
            if (i == _hiddenTile)
            {
                //_hiddenTile = -1;
                it->second->setSingleChildOn(1);
            }
	
	    if (i!=_curTile)
	    {
		 if (i == _hiddenTile) 
		 _aserver->createSynth("SoundFile_Event_Stereo", 
				_aserver->nextNodeId(), _hiddenTileArgs);
	  	 else 
		 _aserver->createSynth("SoundFile_Event_Stereo", 
				   _aserver->nextNodeId(), _regTileArgs);
	 
                 _curTile=i;	
            }
        }
        // Unoccupied tile
        else
        {
            it->second->setSingleChildOn(0);
        }
        
        // Hidden tile
        if (0)//i == _hiddenTile)
        {
            it->second->setSingleChildOn(1);
        }
        i++;

    //std::cout << "Position: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
    //std::cout << "Min: " << xmin << " " << ymin << std::endl;
    //std::cout << "Max: " << xmax << " " << ymax << std::endl;
    }
    
    //std::cout << "Position: " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
    //std::cout << "Min: " << xmin << " " << ymin << std::endl;
    //std::cout << "Max: " << xmax << " " << ymax << std::endl;
}

void WaterMaze::menuCallback(MenuItem * item)
{
    if(item == _loadButton)
    {

    }

    else if (item == _clearButton)
    {

    }

    else if (item == _newTileButton)
    {
        newHiddenTile();
    }

    else if (item == _resetButton)
    {
	// Start timer sound
	// float startTime = PluginHelper::getProgramDuration();
	std::cout << "Start timer sound" << std::endl;
    }

    else if (item == _gridCB)
    {
        if (_gridCB->getValue())
        {
            _gridSwitch->setAllChildrenOn();
        }
        else
        {
            _gridSwitch->setAllChildrenOff();
        }
    }
}

bool WaterMaze::processEvent(InteractionEvent * event)
{
    return false;

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();

    if (tie)
    {
        if(tie->getHand() == 0 && tie->getButton() == 0)
        {
            if (tie->getInteraction() == BUTTON_DOWN)
            {
                return true;
            }
            else if (tie->getInteraction() == BUTTON_DRAG)
            {
                return true;
            }
            else if (tie->getInteraction() == BUTTON_UP)
            {
                return true;
            }
            return true;
        }
    }
    return true;
}

void WaterMaze::clear()
{
    PluginHelper::getObjectsRoot()->removeChild(_geoRoot);
}

void WaterMaze::reset()
{

}

void WaterMaze::newHiddenTile()
{
    _hiddenTile = -1;
}

/*** Server Functions ***/
/*
void WaterMaze::connectToServer()
{

}

int WaterMaze::init_SPP(int port)
{
    char com[100];

    ftStatus = FT_Open(0, &ftHandle);
    if (ftStatus != 0)
    {
        std::cout << "FT_Open failed. Error " << ftStatus << std::endl;
        return -1;
    }
    FT_SetBaudRate(ftHandle, 57600);
    FT_SetDataCharacteristics(ftHandle, FT_BITS_8, FT_STOP_BITS_1, FT_PARITY_NONE);
    FT_SetFlowControl(ftHandle, FT_FLOW_NONE, 0, 0);
    FT_SetLatencyTimer(ftHandle, 2);

    std::cout << "Connected to FTDI device." << std::endl;
    _sppConnected = true;
    return 0;
}

void WaterMaze::close_SPP()
{
    if (!_sppConnected)
        return;

    FT_Close (ftHandle);
}

void WaterMaze::write_SPP(int bytes, unsigned char* buf)
{
    if (!_sppConnected)
        return;
   
    std::cout << "Writing " << buf[0] << std::endl;

    DWORD BytesReceived;
    DWORD bytesToWrite = 1;
    int value;
    value = FT_Write(ftHandle, buf, bytesToWrite, &BytesReceived);
    int a = BytesReceived;

    return;
}
*/
};

