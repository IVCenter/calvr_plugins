#include "WaterMaze.h"

using namespace cvr;
using namespace osg;
using namespace std;

namespace WaterMaze
{

WaterMaze * WaterMaze::_myPtr = NULL;

CVRPLUGIN(WaterMaze)

WaterMaze::WaterMaze()
{
    _myPtr = this;
    _geoRoot = new osg::MatrixTransform();

    _heightOffset = ConfigManager::getFloat("value", 
        "Plugin.WaterMaze.StartingHeight", 300.0);

//    osg::Matrixd mat;
//    mat.makeTranslate(0, -3000, -_heightOffset);
//   _geoRoot->setMatrix(mat);
    PluginHelper::getObjectsRoot()->addChild(_geoRoot);
    _loaded = false;
}

WaterMaze::~WaterMaze()
{
}

WaterMaze * WaterMaze::instance()
{
    return _myPtr;
}

bool WaterMaze::init()
{
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


    _positionMenu = new SubMenu("Reset position");
    _WaterMazeMenu->addItem(_positionMenu);

    for (int i = 0; i < 4; ++i)
    {
        char buffer[50];
        sprintf(buffer, "Corner %d", i + 1);

        MenuButton * button = new MenuButton(buffer);
        button->setCallback(this);
        _positionMenu->addItem(button);

        _positionButtons.push_back(button);
    }

    MenuButton * button = new MenuButton("Center");
    button->setCallback(this);
    _positionMenu->addItem(button);
    _positionButtons.push_back(button);
    
    _detailsMenu = new SubMenu("Levels of Detail");
    _WaterMazeMenu->addItem(_detailsMenu);

    _wallColorCB = new MenuCheckbox("Colored Walls", true);
    _wallColorCB->setCallback(this);
    _detailsMenu->addItem(_wallColorCB);

    _shapesCB = new MenuCheckbox("Shapes", true);
    _shapesCB->setCallback(this);
    _detailsMenu->addItem(_shapesCB);

    _furnitureCB = new MenuCheckbox("Furniture", true);
    _furnitureCB->setCallback(this);
    _detailsMenu->addItem(_furnitureCB);

    _lightingCB = new MenuCheckbox("Lighting", true);
    _lightingCB->setCallback(this);
    _detailsMenu->addItem(_lightingCB);

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


    widthTile = ConfigManager::getFloat("value", "Plugin.WaterMaze.WidthTile", 2000.0);
    heightTile = ConfigManager::getFloat("value", "Plugin.WaterMaze.HeightTile", 2000.0);
    numWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumWidth", 10.0);
    numHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.NumHeight", 10.0);
    depth = ConfigManager::getFloat("value", "Plugin.WaterMaze.Depth", 10.0);
    wallHeight = ConfigManager::getFloat("value", "Plugin.WaterMaze.WallHeight", 2500.0);
    gridWidth = ConfigManager::getFloat("value", "Plugin.WaterMaze.GridWidth", 5.0);

    chooseNewTile();

    return true;
}

void WaterMaze::load()
{
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
            
            // Save four corners and center for starting positions

            // bottom left
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
            // top left 
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
            // bottom right
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
            // top right
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

    // center position
    osg::MatrixTransform * tileMat = new osg::MatrixTransform();
    osg::Matrixd transMat;
    
    transMat.makeTranslate(osg::Vec3(-(widthTile*numWidth*0.5), 
                                -(heightTile*numHeight*0.5),
                                -_heightOffset));

    tileMat->setMatrix(transMat);
    _tilePositions.push_back(tileMat);

    
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
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
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
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
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
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
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
    _wallColorSwitch->addChild(geode);

    sd = new osg::ShapeDrawable(box);
    sd->setColor(osg::Vec4(1.0, 1.0, 1.0, 1));
    geode = new osg::Geode();
    geode->addDrawable(sd);
    geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
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




    // furniture

    osg::Node *painting, *desertpainting, *bookshelf, *chair;
    std::string dataDir = ConfigManager::getEntry("Plugin.WaterMaze.DataDir");

    painting = osgDB::readNodeFile(dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Painting"));
    if (painting)
    {
        osg::PositionAttitudeTransform * pat = new osg::PositionAttitudeTransform();
        float scale = 6.0;
        pat->setScale(osg::Vec3(scale, scale, scale));
        pat->setAttitude(osg::Quat(M_PI/2, osg::Vec3(0, 0, 1)));
        pat->setPosition(osg::Vec3(-widthTile*.75, (numHeight-2) * heightTile/2, wallHeight/3));
        pat->addChild(painting);
        _furnitureSwitch->addChild(pat);
    }

    desertpainting = osgDB::readNodeFile(dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Clock"));
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

    bookshelf = osgDB::readNodeFile(dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Bookshelf"));
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

    chair = osgDB::readNodeFile(dataDir + ConfigManager::getEntry("Plugin.WaterMaze.Models.Chair"));
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


    _loaded = true;

    osg::Matrixd mat;
    mat = _tilePositions[0]->getMatrix();
    PluginHelper::setObjectMatrix(mat);
}

void WaterMaze::preFrame()
{
    if (_hiddenTile < 0)
    {

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
        if (!_loaded)
            load();

        PluginHelper::getObjectsRoot()->addChild(_geoRoot);
    }

    else if (item == _clearButton)
    {
        if (!_loaded)
            return;

        PluginHelper::getObjectsRoot()->removeChild(_geoRoot);
    }

    else if (item == _newTileButton)
    {
        chooseNewTile();
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

    else if (item == _wallColorCB)
    {
        if (_wallColorCB->getValue())
        {
            _wallColorSwitch->setAllChildrenOn();
            _wallWhiteSwitch->setAllChildrenOff();
        }
        else
        {
            _wallColorSwitch->setAllChildrenOff();
            _wallWhiteSwitch->setAllChildrenOn();
        }
    }

    else if (item == _shapesCB)
    {
        if (_shapesCB->getValue())
        {
            _shapeSwitch->setAllChildrenOn();
        }
        else
        {
            _shapeSwitch->setAllChildrenOn();
        }
    }

    else if (item == _furnitureCB)
    {
        if (_furnitureCB->getValue())
        {
            _furnitureSwitch->setAllChildrenOn();
        }
        else
        {
            _furnitureSwitch->setAllChildrenOff();
        }
    }

    else if (item == _lightingCB)
    {
        if (_lightingCB->getValue())
        {

        }
        else
        {

        }
    }

    int i = 0;
    for (std::vector<MenuButton*>::iterator it = _positionButtons.begin();
         it != _positionButtons.end(); ++it)
    {
        if ((*it) == item)
        {
            std::cout << (*it)->getText() << std::endl;

/*            osg::Matrixd mat, multMat;

            osg::Vec3 newPos =  _tilePositions[i]->getPosition() * _geoRoot->getInverseMatrix();
            osg::Vec3 origPos = PluginHelper::getObjectMatrix().getTrans();
            osg::Vec3 transVec = newPos - origPos;

            std::cout << "newPos = " << newPos[0] << " " << newPos[1] << " " << newPos[2] << std::endl;
            std::cout << "origPos = " << origPos[0] << " " << origPos[1] << " " << origPos[2] << std::endl;
            std::cout << "transVec = " << transVec[0] << " " << transVec[1] << " " << transVec[2] << std::endl;

            mat = PluginHelper::getObjectMatrix();
            multMat.makeTranslate(transVec[0], transVec[1], 0);

            //mat.setTrans(multMat * );
            mat = multMat * mat; */
            
            osg::Matrixd mat;
            mat = _tilePositions[i]->getMatrix();
            PluginHelper::setObjectMatrix(mat);
        }
        ++i;
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
    _loaded = false;
}

void WaterMaze::reset()
{

}

void WaterMaze::chooseNewTile()
{
    _hiddenTile = rand() % (int)(numWidth * numHeight);
    std::cout << "Hidden tile = " << _hiddenTile << std::endl;
}

};

