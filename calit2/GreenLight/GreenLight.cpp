// John Mangan (Summer 2011)
// Redone BlackBox plugin from Covise

#include "GreenLight.h"

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
    delete _glMenu;
    if (_showSceneCheckbox) delete _showSceneCheckbox;

    vector<Entity *>::iterator vit;
    for (vit = _door.begin(); vit != _door.end(); vit++)
    {
        delete *vit;
    }
    _door.clear();

    cerr << "GreenLight destroyed." << endl;
}

bool GreenLight::init()
{
    cerr << "GreenLight init()." << endl;

    /*** Menu Setup ***/
    _glMenu = new SubMenu("GreenLight","GreenLight");
    _glMenu->setCallback(this);
    PluginHelper::addRootMenuItem(_glMenu);

    _showSceneCheckbox = new MenuCheckbox("Load Box",false);
    _showSceneCheckbox->setCallback(this);
    _glMenu->addItem(_showSceneCheckbox);

    _displayComponentsMenu = NULL;
    _displayFrameCheckbox = NULL;
    _displayDoorsCheckbox = NULL;
    _displayWaterPipesCheckbox = NULL;
    _displayElectricalCheckbox = NULL;
    _displayFansCheckbox = NULL;
    /*** End Menu Setup ***/

    /*** Entity Defaults ***/
    _box = NULL;
    _waterPipes = NULL;
    _electrical = NULL;
    _fans = NULL;
    /*** End Entity Defaults ***/

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
}

void GreenLight::preFrame()
{
    for (int d = 0; d < _door.size(); d++)
        _door[d]->handleAnimation();
}

void GreenLight::postFrame()
{
}

bool GreenLight::keyEvent(bool keyDown, int key, int mod)
{
//    cerr << "GreenLight keyEvent: keyDown: " << keyDown << " key: " << key << " char: " << (char)key << " mod: " << mod << endl;
    if (keyDown && key == 65361) // left arrow
    {
    }
    else if (keyDown && key == 65362) // up arrow
    {
    }
    else if (keyDown && key == 65363) // right arrow
    {
    }
    else if (keyDown && key == 65364) // down arrow
    {
    }
    else if (keyDown && key == 'p')
    {
    }    

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
