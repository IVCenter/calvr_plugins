// John Mangan (Summer 2011)
// Plugin for CalVR -- GreenLight Project
// Many models taked from prior Covise Plugin (BlackBoxInfo)

#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <kernel/PluginHelper.h>
#include <kernel/InteractionManager.h>

CVRPLUGIN(GreenLight)

GreenLight::GreenLight()
{
    cerr << "GreenLight created." << endl;
}

GreenLight::~GreenLight()
{
    if (_glMenu) delete _glMenu;
    if (_showSceneCheckbox) delete _showSceneCheckbox;
    if (_displayComponentsMenu) delete _displayComponentsMenu;
    if (_componentsViewCheckbox) delete _componentsViewCheckbox;
    if (_displayFrameCheckbox) delete _displayFrameCheckbox;
    if (_displayDoorsCheckbox) delete _displayDoorsCheckbox;
    if (_displayWaterPipesCheckbox) delete _displayWaterPipesCheckbox;
    if (_displayElectricalCheckbox) delete _displayElectricalCheckbox;
    if (_displayFansCheckbox) delete _displayFansCheckbox;
    if (_displayRacksCheckbox) delete _displayRacksCheckbox;
    if (_powerMenu) delete _powerMenu;
    if (_displayPowerCheckbox) delete _displayPowerCheckbox;
    if (_loadPowerButton) delete _loadPowerButton;

    if (_box) delete _box;
    if (_waterPipes) delete _waterPipes;
    if (_electrical) delete _electrical;
    if (_fans) delete _fans;

    vector<Entity *>::iterator vit;
    for (vit = _door.begin(); vit != _door.end(); vit++)
    {
        if (*vit) delete *vit;
    }
    _door.clear();

    for (vit = _rack.begin(); vit != _rack.end(); vit++)
    {
        if (*vit) delete *vit;
    }
    _rack.clear();

    cerr << "GreenLight destroyed." << endl;
}

bool GreenLight::init()
{
    cerr << "GreenLight init()." << endl;

    /*** Menu Setup ***/
    _glMenu = new SubMenu("GreenLight","GreenLight");
    _glMenu->setCallback(this);
    PluginHelper::addRootMenuItem(_glMenu);

    _showSceneCheckbox = new MenuCheckbox("Load Scene",false);
    _showSceneCheckbox->setCallback(this);
    _glMenu->addItem(_showSceneCheckbox);

    _displayComponentsMenu = NULL;
    _componentsViewCheckbox = NULL;
    _displayFrameCheckbox = NULL;
    _displayDoorsCheckbox = NULL;
    _displayWaterPipesCheckbox = NULL;
    _displayElectricalCheckbox = NULL;
    _displayFansCheckbox = NULL;
    _displayRacksCheckbox = NULL;

    _powerMenu = NULL;
    _displayPowerCheckbox = NULL;
    _loadPowerButton = NULL;
    /*** End Menu Setup ***/

    /*** Entity Defaults ***/
    _box = NULL;
    _waterPipes = NULL;
    _electrical = NULL;
    _fans = NULL;
    /*** End Entity Defaults ***/

    downloadFile(ConfigManager::getEntry("download", "Plugin.GreenLight.Hardware", ""),
                 ConfigManager::getEntry("local", "Plugin.GreenLight.Hardware", ""),
                 _hardwareContents);

    return true;
}

void GreenLight::menuCallback(MenuItem * item)
{
    if (item == _showSceneCheckbox)
    {
        // Load as neccessary
        if (!_box)
        {
            if (loadScene())
                _showSceneCheckbox->setText("Show Scene");
            else
            {
                cerr << "Error: loadScene() failed." << endl;
                _showSceneCheckbox->setValue(false);
                return;
            }

        }

        if (_showSceneCheckbox->getValue())
            PluginHelper::getObjectsRoot()->addChild(_box->transform);
        else
            PluginHelper::getObjectsRoot()->removeChild(_box->transform);
    }
    else if (item == _componentsViewCheckbox)
    {
        bool transparent = _componentsViewCheckbox->getValue();
        _box->setTransparency(transparent);
        _waterPipes->setTransparency(transparent);
        _electrical->setTransparency(transparent);
        _fans->setTransparency(transparent);
        for (int d = 0; d < _door.size(); d++)
            _door[d]->setTransparency(transparent);
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->setTransparency(transparent);
    }
    else if (item == _displayFrameCheckbox)
    {
        _box->showVisual(_displayFrameCheckbox->getValue());
    }
    else if (item == _displayDoorsCheckbox)
    {
        for (int d = 0; d < _door.size(); d++)
            _door[d]->showVisual(_displayDoorsCheckbox->getValue());
    }
    else if (item == _displayWaterPipesCheckbox)
    {
        _waterPipes->showVisual(_displayWaterPipesCheckbox->getValue());
    }
    else if (item == _displayElectricalCheckbox)
    {
        _electrical->showVisual(_displayFrameCheckbox->getValue());
    }
    else if (item == _displayFansCheckbox)
    {
        _fans->showVisual(_displayFansCheckbox->getValue());
    }
    else if (item == _displayRacksCheckbox)
    {
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->showVisual(_displayRacksCheckbox->getValue());
    }
    else if (item == _loadPowerButton)
    {
        downloadFile(ConfigManager::getEntry("download", "Plugin.GreenLight.Power", ""),
                     ConfigManager::getEntry("local", "Plugin.GreenLight.Power", ""),
                     _powerContents);

        if (!_displayPowerCheckbox)
        {
            ifstream file;
            file.open(ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "").c_str());
            if (file)
            {
                _displayPowerCheckbox = new MenuCheckbox("Display Power Consumption",false);
                _displayPowerCheckbox->setCallback(this);
                _powerMenu->addItem(_displayPowerCheckbox);
            }
            file.close();
        }

    }
    else if (item == _displayPowerCheckbox)
    {
//_box->setColor(Vec3(1,1,0));
//for(int r = 0; r < _rack.size(); r++)
//_rack[r]->setColor(Vec3(1,0,0));
        setPowerColors(_displayPowerCheckbox->getValue());
    }
}

void GreenLight::preFrame()
{
    for (int d = 0; d < _door.size(); d++)
        _door[d]->handleAnimation();
    for (int r = 0; r < _rack.size(); r++)
        _rack[r]->handleAnimation();
}

void GreenLight::postFrame()
{
}

bool GreenLight::keyEvent(bool keyDown, int key, int mod)
{
//    cerr << "GreenLight keyEvent: keyDown: " << keyDown << " key: " << key << " char: " << (char)key << " mod: " << mod << endl;
    return false;
}

bool GreenLight::buttonEvent(int type, int button, int hand, const osg::Matrix& mat)
{
/*
    cerr << "Button event type: ";
    switch(type)
    {
        case BUTTON_DOWN:
            cerr << "BUTTON_DOWN ";
            break;
        case BUTTON_UP:
            cerr << "BUTTON_UP ";
            break;
        case BUTTON_DRAG:
            cerr << "BUTTON_DRAG ";
            break;
        case BUTTON_DOUBLE_CLICK:
            cerr << "BUTTON_DOUBLE_CLICK ";
            break;
        default:
            cerr << "UNKNOWN ";
            break;
    }

    cerr << "hand: " << hand << " button: " << button << endl;
*/

    if (type != BUTTON_DOWN || button != 0)
        return false;

    if (!_box)
        return false;

    // process intersection
    osg::Vec3 pointerStart, pointerEnd;
    std::vector<IsectInfo> isecvec;

    pointerStart = mat.getTrans();
    pointerEnd.set(0.0f, 10000.0f, 0.0f);
    pointerEnd = pointerEnd * mat;

    isecvec = getObjectIntersection(PluginHelper::getScene(),
                pointerStart, pointerEnd);

    if (isecvec.size() > 0)
        return handleIntersection(isecvec[0].geode);

    return false;
}

bool GreenLight::mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat)
{
/*
    cerr << "Mouse Button event type: ";
    switch(type)
    {
        case MOUSE_BUTTON_DOWN:
            cerr << "MOUSE_BUTTON_DOWN ";
            break;
        case MOUSE_BUTTON_UP:
            cerr << "MOUSE_BUTTON_UP ";
            break;
        case MOUSE_DRAG:
            cerr << "MOUSE_DRAG ";
            break;
        case MOUSE_DOUBLE_CLICK:
            cerr << "MOUSE_DOUBLE_CLICK ";
            break;
        default:
            cerr << "UNKNOWN ";
            break;
    }

    cerr << "button: " << button << endl;
*/
    // Left Button Click
    if (type != MOUSE_BUTTON_DOWN || button != 0)
        return false;

    if (!_box)
        return false;

    // process mouse intersection
    osg::Vec3 pointerStart, pointerEnd;
    std::vector<IsectInfo> isecvec;

    pointerStart = mat.getTrans();
    pointerEnd.set(0.0f, 10000.0f, 0.0f);
    pointerEnd = pointerEnd * mat;

    isecvec = getObjectIntersection(PluginHelper::getScene(),
                pointerStart, pointerEnd);

    if (isecvec.size() > 0)
        return handleIntersection(isecvec[0].geode);

    return false;
}
