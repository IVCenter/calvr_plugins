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
   if (ConfigManager::getEntry("Plugin.WaterMaze.Sound") == "on")
   {
       _aserver = new SCServer(name, "127.0.0.1" , "57110", synthdir);
       _regTileBuf = new Buffer(_aserver, _aserver->nextBufferNum());
       _hiddenTileBuf = new Buffer(_aserver, _aserver->nextBufferNum());
       _aserver->dumpOSC(1);
   }

   _curTile=0;



    //_sppConnected = false;
    _hiddenTile = -1;

    

    float heightOffset = ConfigManager::getFloat("value", 
        "Plugin.WaterMaze.StartingHeight", 300.0);

    osg::Matrixd mat;
    mat.makeTranslate(0, -3000, -heightOffset);
    _geoRoot->setMatrix(mat);
    PluginHelper::getObjectsRoot()->addChild(_geoRoot);
    
    _loaded = false;
    _currentTrial = 0;
    _startTime = 0;
    _lastTimeLeft = 0;
    _resetTime = true;
    _fileTick = .5;

    std::cout << "Welcome to WaterMaze!\n" <<
            "l - load geometry\n" <<
            "n - next trial\n" <<
            "r - repeat trial\n" <<
            "b - back to previous trial\n" <<
            "p - play/pause\n" << 
            "h - help\n" << 
            "1-9 - reset position" << std::endl;
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
    if (ConfigManager::getEntry("Plugin.WaterMaze.Sound") == "on")
    {
        //_regTileBuf->allocRead("/Users/demo/workspace/svn/libSCresources/projects/watermaze/step.aiff");
        _regTileBuf->allocRead("/Users/administrator/libSCresources/projects/watermaze/step.aiff");

       // _hiddenTileBuf->allocRead("/Users/demo/workspace/svn/libSCresources/projects/watermaze/groove.aiff");
          _hiddenTileBuf->allocRead("/Users/administrator/libSCresources/projects/watermaze/groove.aiff");
    }


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
    //numWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumWidth", 10.0);
    //numHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumHeight", 10.0);
    depth = ConfigManager::getFloat("value", "Plugin.WaterMaze.Depth", 10.0);
    wallHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.WallHeight", 2500.0);
    gridWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.GridWidth", 5.0);

    _dataDir = ConfigManager::getEntry("Plugin.WaterMaze.DataDir");

    std::vector<std::string> trialNames;
    ConfigManager::getChildren("Plugin.WaterMaze.Trials", trialNames);
    for (int i = 0; i < trialNames.size(); ++i)
    {
        float width, height, time;
        width = ConfigManager::getFloat("NumWidth", "Plugin.WaterMaze.Trials." + 
            trialNames[i]);
        height = ConfigManager::getFloat("NumHeight", "Plugin.WaterMaze.Trials." + 
            trialNames[i]);
        time = ConfigManager::getFloat("Time", "Plugin.WaterMaze.Trials." + 
            trialNames[i]);

        Trial trial;
        trial.width = width;
        trial.height = height;
        trial.timeSec = time;
        _trials.push_back(trial);
    }

    for (int i = 0; i < _trials.size(); ++i)
    {

    }

    _lastTimeLeft = _trials[_currentTrial].timeSec;

    newHiddenTile();

    return true;
}

void WaterMaze::load(int numWidth, int numHeight)
{
    _loaded = false;

    // Set up models

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
            

            // Save corner positions
            if (i == 0 && j == 0)
            {
                osg::MatrixTransform * tileMat = new osg::MatrixTransform();
                osg::Matrixd mat, rotMat, transMat;

                mat.makeTranslate(osg::Vec3(0,0,0));
                rotMat.makeRotate(M_PI/4, osg::Vec3(0, 0, 1));
                transMat.makeTranslate((-tilePat->getPosition() +
                osg::Vec3(-widthTile, -heightTile, -_heightOffset)));

                mat.preMult(rotMat);
                mat.preMult(transMat);

                tileMat->setMatrix(mat);
                _tilePositions.push_back(tileMat);
            }
            else if (i == 0 && j == numHeight - 1)
            {
                osg::MatrixTransform * tileMat = new osg::MatrixTransform();
                osg::Matrixd mat, rotMat, transMat;

                mat.makeTranslate(osg::Vec3(0,0,0));
                rotMat.makeRotate(3*M_PI/4, osg::Vec3(0, 0, 1));
                transMat.makeTranslate((-tilePat->getPosition()  +
                osg::Vec3(-widthTile, heightTile, -_heightOffset)));

                mat.preMult(rotMat);
                mat.preMult(transMat);

                tileMat->setMatrix(mat);
                _tilePositions.push_back(tileMat);
            }
            else if (i == numWidth - 1 && j == 0)
            {
                osg::MatrixTransform * tileMat = new osg::MatrixTransform();
                osg::Matrixd mat, rotMat, transMat;

                mat.makeTranslate(osg::Vec3(0,0,0));
                rotMat.makeRotate(7*M_PI/4, osg::Vec3(0, 0, 1));
                transMat.makeTranslate((-tilePat->getPosition() +
                osg::Vec3(widthTile, -heightTile, -_heightOffset)));

                mat.preMult(rotMat);
                mat.preMult(transMat);

                tileMat->setMatrix(mat);
                _tilePositions.push_back(tileMat);
            }
            else if (i == numWidth - 1 && j == numHeight - 1)
            {
                osg::MatrixTransform * tileMat = new osg::MatrixTransform();
                osg::Matrixd mat, rotMat, transMat;

                mat.makeTranslate(osg::Vec3(0,0,0));
                rotMat.makeRotate(5*M_PI/4, osg::Vec3(0, 0, 1));
                transMat.makeTranslate((-tilePat->getPosition() +
                osg::Vec3(widthTile, heightTile, -_heightOffset)));

                mat.preMult(rotMat);
                mat.preMult(transMat);

                tileMat->setMatrix(mat);
                _tilePositions.push_back(tileMat);
            }

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
    
    _wallWhiteSwitch = new osg::Switch();
    _wallColorSwitch = new osg::Switch();
    _shapeSwitch = new osg::Switch();
    _furnitureSwitch = new osg::Switch();

    std::string wallTex = ConfigManager::getEntry("Plugin.WaterMaze.Textures.Wall");
    osg::Texture2D* tex = new osg::Texture2D();
    osg::ref_ptr<osg::Image> img;

    img = osgDB::readImageFile(_dataDir + wallTex);
    if (img)
    {
        tex->setImage(img);
        tex->setResizeNonPowerOfTwoHint(false);
        tex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
        tex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    }



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
     
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallWhiteSwitch->addChild(geode);

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
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallWhiteSwitch->addChild(geode);

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
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallWhiteSwitch->addChild(geode);

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
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    _wallWhiteSwitch->addChild(geode);

    _geoRoot->addChild(_wallColorSwitch);
    _geoRoot->addChild(_wallWhiteSwitch);
    
    _wallColorSwitch->setAllChildrenOn();
    _wallWhiteSwitch->setAllChildrenOff();


    // ceiling
    pos = osg::Vec3((numWidth-2) * widthTile * .5, 
                    (numHeight-2) * heightTile * .5, 
                    wallHeight);
    box = new osg::Box(pos, numWidth * widthTile, numHeight * heightTile, 5);
    sd = new osg::ShapeDrawable(box);
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
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
    
    if (ConfigManager::getEntry("Plugin.WaterMaze.Sound") == "on")
    {
        if (_aserver)
            _aserver->dumpOSC(0);
    }

    osg::Node *painting, *desertpainting, *bookshelf, *chair;


    painting = osgDB::readNodeFile(_dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Painting"));
    if (painting)
    {
    std::cout << "Painting loaded" << std::endl;
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();
        float scale = 6.0;
        pat->setScale(osg::Vec3(scale, scale, scale));
        pat->setAttitude(osg::Quat(M_PI/2, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Vec3(-widthTile*.75, (numHeight-2) * heightTile/2, wallHeight/3));
        pat->addChild(painting);
        _furnitureSwitch->addChild(pat);
    }

    desertpainting = osgDB::readNodeFile(_dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Clock"));
    if (desertpainting)
    {
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();
        float scale = 12.0;
        pat->setScale(osg::Vec3(scale, scale, scale));
        pat->setAttitude(osg::Quat(M_PI/2, osg::Vec3(1, 0, 0),
        0,      osg::Vec3(0, 1, 0),
        -M_PI/2, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Vec3(widthTile * (numWidth-2), (numHeight-2) * heightTile/2, wallHeight/3));
        pat->addChild(desertpainting);
        _furnitureSwitch->addChild(pat);
    }

    bookshelf = osgDB::readNodeFile(_dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Bookshelf"));
    if (bookshelf)
    {
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();
        float scale = 6.0;
        pat->setScale(osg::Vec3(scale, scale, scale));
        pat->setPosition(osg::Vec3(widthTile * (numWidth-2) * 0.5,
        ((numHeight-2) * heightTile) + 0.5*heightTile,
        0));
        pat->addChild(bookshelf);
        _furnitureSwitch->addChild(pat);
    }

    chair = osgDB::readNodeFile(_dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Chair"));
    if (chair)
    {
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();
        float scale = 6.0;
        pat->setScale(osg::Vec3(scale, scale, scale));
        pat->setAttitude(osg::Quat(3*M_PI/4, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Vec3(-widthTile/2, -heightTile/2, 0));
        pat->addChild(chair);
        _furnitureSwitch->addChild(pat);
   }

  _geoRoot->addChild(_furnitureSwitch);

  osg::Matrixd mat;
  mat = _tilePositions[0]->getMatrix();
  PluginHelper::setObjectMatrix(mat);

  _loaded = true;

}


void WaterMaze::preFrame()
{
    if (!_loaded)
        return;

    if (ConfigManager::getEntry("Plugin.WaterMaze.Sound") == "on")
    {
        _regTileArgs["bufnum"] = _regTileBuf->getBufNum();
        _hiddenTileArgs["bufnum"] = _hiddenTileBuf->getBufNum();
    }


    if (_runningTrial && 
        PluginHelper::getProgramDuration() - _startTime > _trials[_currentTrial].timeSec)
    {
        std::cout << "Trial complete!  Press 'n' to continue to next trial." << std::endl;
        _runningTrial = false;
        _resetTime = true;
    }

    if (!_runningTrial)
        return;

    int timeLeft;
    timeLeft = _trials[_currentTrial].timeSec - 
        ((int)(PluginHelper::getProgramDuration() - _startTime));
    if (timeLeft < _lastTimeLeft)
    {
        _lastTimeLeft = timeLeft;
        std::cout << timeLeft << " seconds left." << std::endl;
    }




    if (PluginHelper::getProgramDuration() - _fileTimer > _fileTick)
    {
        ofstream outFile;
        outFile.open(_dataFile.c_str(), ios::app);

        if (outFile) 
        {
            float elapsedTime = PluginHelper::getProgramDuration() - _startTime;
            osg::Vec3 pos = PluginHelper::getHeadMat(0).getTrans();
            outFile << elapsedTime << " " << pos.x() << " " << pos.y() << " " << endl;
            outFile << std::endl;
            outFile << flush;
            outFile.close();

            _fileTimer = PluginHelper::getProgramDuration();
        }

        else
        {
            cout << "WaterMaze: Unable to open file " << _dataFile << endl;
        }
    }




    osg::Vec3 pos = osg::Vec3(0,0,0) * cvr::PluginHelper::getHeadMat() * 
        PluginHelper::getWorldToObjectTransform() * _geoRoot->getInverseMatrix();

    osg::Vec3 bottomLeft, topRight;
    bottomLeft = osg::Vec3(0,0,0) * _geoRoot->getMatrix();
    topRight = osg::Vec3(widthTile * _trials[_currentTrial].width, 
                         heightTile * _trials[_currentTrial].height, 0) * 
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
            if (it->second)
            {
                it->second->setSingleChildOn(2);
                if (i == _hiddenTile)
                {
                    //_hiddenTile = -1;
                    it->second->setSingleChildOn(1);
                }
            }

            if (ConfigManager::getEntry("Plugin.WaterMaze.Sound") == "on")
            {
                if (i != _curTile)
                {
                    if (i == _hiddenTile && _aserver)
                    {
                        _aserver->createSynth("SoundFile_Event_Stereo", 
                        _aserver->nextNodeId(), _hiddenTileArgs);
                    }
                    else if (_aserver)
                    {
                        _aserver->createSynth("SoundFile_Event_Stereo", 
                            _aserver->nextNodeId(), _regTileArgs);
                        _curTile=i;	
                    }
                }
            }
        }
        // Unoccupied tile
        else
        {
            if (it->second)
                it->second->setSingleChildOn(0);
        }
        
        // Hidden tile
        if (0)//i == _hiddenTile)
        {
            if (it->second)
                it->second->setSingleChildOn(1);
        }
        i++;
    }
}

void WaterMaze::menuCallback(MenuItem * item)
{
    if(item == _loadButton)
    {
        if (!_loaded)
            load(_trials[0].width, _trials[0].height);
    }

    else if (item == _clearButton)
    {
        clear();
    }

    else if (item == _newTileButton)
    {
        newHiddenTile();
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
    KeyboardInteractionEvent * kie = event->asKeyboardEvent();
    if (kie)
    {
        if (kie->getInteraction() == KEY_UP && kie->getKey() == 'n')
        {
            // next trial
            if (_runningTrial)
            {
                return true;
            }

            if (_currentTrial == _trials.size() - 1)
            {
                std::cout << "No more trials." << std::endl;
                return true;
            }

            _currentTrial++;
            
            std::cout << "Loading next trial." << std::endl;
            std::cout << _trials[_currentTrial].width << " x " << 
                _trials[_currentTrial].height << ", " << _trials[_currentTrial].timeSec
                << " seconds." << std::endl;

            clear();
            load(_trials[_currentTrial].width, _trials[_currentTrial].height); 

            return true;
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 'r')
        {
            // repeat last trial
            if (_runningTrial)
            {
                return true;
            }

            std::cout << "Repeating last trial." << std::endl;
            std::cout << _trials[_currentTrial].width << " x " << 
                _trials[_currentTrial].height << ", " << _trials[_currentTrial].timeSec
                << " seconds." << std::endl;

            return true;
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 'b')
        {
            // back to previous trial
            if (_runningTrial)
            {
                return true;
            }

            if (_currentTrial - 1 < 0)
            {
                std::cout << "No previous trials." << std::endl;
                return true;
            }

            _currentTrial--;
            
            std::cout << "Back to previous trial." << std::endl;
            std::cout << _trials[_currentTrial].width << " x " << 
                _trials[_currentTrial].height << ", " << _trials[_currentTrial].timeSec
                << " seconds." << std::endl;

            clear();
            load(_trials[_currentTrial].width, _trials[_currentTrial].height); 
            
            return true;
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 'p')
        {
            // pause/play
            if (_runningTrial)
            {
                std::cout << "Paused." << std::endl; 
                _runningTrial = false;
            }
            else
            {
                std::cout << "Play." << std::endl; 
                _runningTrial = true;

                if (_resetTime)
                {
                    _startTime = PluginHelper::getProgramDuration();
                    _resetTime = false;
                    _lastTimeLeft = _trials[_currentTrial].timeSec;


                    // Setup write to data file
                    time_t timet;
                    time(&timet);
                    char buf[100];

                    sprintf(buf, "/Logs/%dx%d-%dsec-%ld.txt", _trials[_currentTrial].width, 
                        _trials[_currentTrial].height, _trials[_currentTrial].timeSec, timet);
                    _dataFile = _dataDir + buf;
                    std::cout << "Recording trial to" << _dataFile << std::endl;

                    _fileTimer = PluginHelper::getProgramDuration();
                }
            }
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 's')
        {
            // start/stop
            if (_runningTrial)
            {
                std::cout << "Stopping trial." << std::endl; 
                _runningTrial = false;
                _resetTime = true;
            }
            else
            {
                std::cout << "Starting trial." << std::endl; 
                _runningTrial = true;
            }
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 'l')
        {
            std::cout << "Loading geometry." << std::endl;
            load(_trials[_currentTrial].width, _trials[_currentTrial].height);
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == 'h')
        {
            std::cout << "Welcome to WaterMaze!\n" <<
            "l - load geometry\n" <<
            "n - next trial\n" <<
            "r - repeat trial\n" <<
            "b - back to previous trial\n" <<
            "p - play/pause\n" << 
            "h - help\n" << 
            "1-9 - reset position" << std::endl;
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == '1')
        {
            if (_runningTrial)
            {
                return true;
            }
            osg::Matrixd mat;
            mat = _tilePositions[0]->getMatrix();
            PluginHelper::setObjectMatrix(mat);
            
            // turn off all colored tiles
            std::map<osg::Vec3, osg::Switch *>::iterator it;
            for (it = _tileSwitches.begin(); it != _tileSwitches.end(); ++it)
            {
                if (it->second)
                {
                    it->second->setSingleChildOn(0);
                }
            }
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == '2')
        {
            if (_runningTrial)
            {
                return true;
            }
            osg::Matrixd mat;
            mat = _tilePositions[1]->getMatrix();
            PluginHelper::setObjectMatrix(mat);

            // turn off all colored tiles
            std::map<osg::Vec3, osg::Switch *>::iterator it;
            for (it = _tileSwitches.begin(); it != _tileSwitches.end(); ++it)
            {
                if (it->second)
                {
                    it->second->setSingleChildOn(0);
                }
            }
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == '3')
        {
            if (_runningTrial)
            {
                return true;
            }
            osg::Matrixd mat;
            mat = _tilePositions[2]->getMatrix();
            PluginHelper::setObjectMatrix(mat);

            // turn off all colored tiles
            std::map<osg::Vec3, osg::Switch *>::iterator it;
            for (it = _tileSwitches.begin(); it != _tileSwitches.end(); ++it)
            {
                if (it->second)
                {
                    it->second->setSingleChildOn(0);
                }
            }
        }

        else if (kie->getInteraction() == KEY_UP && kie->getKey() == '4')
        {
            if (_runningTrial)
            {
                return true;
            }
            osg::Matrixd mat;
            mat = _tilePositions[3]->getMatrix();
            PluginHelper::setObjectMatrix(mat);

            // turn off all colored tiles
            std::map<osg::Vec3, osg::Switch *>::iterator it;
            for (it = _tileSwitches.begin(); it != _tileSwitches.end(); ++it)
            {
                if (it->second)
                {
                    it->second->setSingleChildOn(0);
                }
            }
        }
    }


    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if (tie)
    {
        if(tie->getHand() == 0 && tie->getButton() == 0)
        {
            if (tie->getInteraction() == BUTTON_DOWN && !_runningTrial)
            {
                return true;
            }
            else if (tie->getInteraction() == BUTTON_DRAG && !_runningTrial)
            {
                return true;
            }
            else if (tie->getInteraction() == BUTTON_UP)
            {
                return false;
            }
            return false;
        }
    }
    return false;
}

void WaterMaze::clear()
{
    _loaded = false;
    PluginHelper::getObjectsRoot()->removeChild(_geoRoot);

    _tileSwitches.clear();
    _tilePositions.clear();

    _geoRoot = new osg::MatrixTransform();

    float heightOffset = ConfigManager::getFloat("value", 
        "Plugin.WaterMaze.StartingHeight", 300.0);
    osg::Matrixd mat;
    mat.makeTranslate(0, -3000, -heightOffset);
    _geoRoot->setMatrix(mat);
    PluginHelper::getObjectsRoot()->addChild(_geoRoot);
}

void WaterMaze::reset()
{

}

void WaterMaze::newHiddenTile()
{
    _hiddenTile = -1;
}

};

