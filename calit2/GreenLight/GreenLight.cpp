// John Mangan (Summer 2011)
// Plugin for CalVR -- GreenLight Project
// Many models taked from prior Covise Plugin (BlackBoxInfo)

#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <kernel/ComController.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>

#include <osgDB/ReadFile>

CVRPLUGIN(GreenLight)

// Static Variables
osg::ref_ptr<osg::Uniform> GreenLight::Component::_displayTexturesUni = new osg::Uniform("showTexture",false);
osg::ref_ptr<osg::Uniform> GreenLight::Component::_neverTextureUni = new osg::Uniform("showTexture",false);

GreenLight::GreenLight()
{
    std::cerr << "GreenLight created." << std::endl;
}

GreenLight::~GreenLight()
{
    if (_glMenu) delete _glMenu;
    if (_showSceneCheckbox) delete _showSceneCheckbox;

    if (_hardwareSelectionMenu) delete _hardwareSelectionMenu;
    if (_selectionModeCheckbox) delete _selectionModeCheckbox;
    if (_selectClusterMenu) delete _selectClusterMenu;

    std::set< cvr::MenuCheckbox * >::iterator chit;
    for (chit = _clusterCheckbox.begin(); chit != _clusterCheckbox.end(); chit++)
    {
        if (*chit) delete *chit;
    }
    _clusterCheckbox.clear();

    if (_selectAllButton) delete _selectAllButton;
    if (_deselectAllButton) delete _deselectAllButton;

    if (_displayComponentsMenu) delete _displayComponentsMenu;
    if (_xrayViewCheckbox) delete _xrayViewCheckbox;
    if (_displayFrameCheckbox) delete _displayFrameCheckbox;
    if (_displayDoorsCheckbox) delete _displayDoorsCheckbox;
    if (_displayWaterPipesCheckbox) delete _displayWaterPipesCheckbox;
    if (_displayElectricalCheckbox) delete _displayElectricalCheckbox;
    if (_displayFansCheckbox) delete _displayFansCheckbox;
    if (_displayRacksCheckbox) delete _displayRacksCheckbox;
    if (_displayComponentTexturesCheckbox) delete _displayComponentTexturesCheckbox;
    if (_powerMenu) delete _powerMenu;
    if (_displayPowerCheckbox) delete _displayPowerCheckbox;
    if (_loadPowerButton) delete _loadPowerButton;
    if (_legendText) delete _legendText;
    if (_legendGradient) delete _legendGradient;
    if (_legendTextOutOfRange) delete _legendTextOutOfRange;
    if (_legendGradientOutOfRange) delete _legendGradientOutOfRange;

    if (_box) delete _box;
    if (_waterPipes) delete _waterPipes;
    if (_electrical) delete _electrical;
    if (_fans) delete _fans;

    std::map< std::string, std::set< Component * > * >::iterator cit;
    for (cit = _cluster.begin(); cit != _cluster.end(); cit++)
    {
        if (cit->second) delete cit->second;
    }
    _cluster.clear();

    std::vector<Entity *>::iterator vit;
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

    std::cerr << "GreenLight destroyed." << std::endl;
}

bool GreenLight::init()
{
    std::cerr << "GreenLight init()." << std::endl;

    /*** Menu Setup ***/
    _glMenu = new cvr::SubMenu("GreenLight","GreenLight");
    _glMenu->setCallback(this);
    cvr::PluginHelper::addRootMenuItem(_glMenu);

    _showSceneCheckbox = new cvr::MenuCheckbox("Load Scene",false);
    _showSceneCheckbox->setCallback(this);
    _glMenu->addItem(_showSceneCheckbox);

    _hardwareSelectionMenu = NULL;
    _selectionModeCheckbox = NULL;
    _selectAllButton = NULL;
    _deselectAllButton = NULL;

    _displayComponentsMenu = NULL;
    _xrayViewCheckbox = NULL;
    _displayFrameCheckbox = NULL;
    _displayDoorsCheckbox = NULL;
    _displayWaterPipesCheckbox = NULL;
    _displayElectricalCheckbox = NULL;
    _displayFansCheckbox = NULL;
    _displayRacksCheckbox = NULL;
    _displayComponentTexturesCheckbox = NULL;

    _powerMenu = NULL;
    _displayPowerCheckbox = NULL;
    _loadPowerButton = NULL;
    _legendText = NULL;
    _legendGradient = NULL;
    _legendTextOutOfRange = NULL;
    _legendGradientOutOfRange = NULL;
    /*** End Menu Setup ***/

    /*** Defaults ***/
    _box = NULL;
    _waterPipes = NULL;
    _electrical = NULL;
    _fans = NULL;

    _shaderProgram = NULL;

    _mouseOver = NULL;
    _wandOver = NULL;
    /*** End Defaults ***/

    return true;
}

void GreenLight::menuCallback(cvr::MenuItem * item)
{
    std::set< cvr::MenuCheckbox * >::iterator chit;

    if (item == _showSceneCheckbox)
    {
        // Load as neccessary
        if (!_box)
        {
            if (!_shaderProgram)
            {
                // First compile shaders
                std::cerr<<"Loading shaders... ";
                _shaderProgram = new osg::Program;

                osg::ref_ptr<osg::Shader> vertShader = new osg::Shader( osg::Shader::VERTEX );
                osg::ref_ptr<osg::Shader> fragShader = new osg::Shader( osg::Shader::FRAGMENT );

                if (utl::loadShaderSource(vertShader, cvr::ConfigManager::getEntry("vertex", "Plugin.GreenLight.Shaders", ""))
                && utl::loadShaderSource(fragShader, cvr::ConfigManager::getEntry("fragment", "Plugin.GreenLight.Shaders", "")))
                {
                    _shaderProgram->addShader( vertShader );
                    _shaderProgram->addShader( fragShader );
                    std::cerr<<"done."<<std::endl;
                }
                else
                    std::cerr<<"failed!"<<std::endl;
                // Done with shaders
            }

            utl::downloadFile(cvr::ConfigManager::getEntry("download", "Plugin.GreenLight.Hardware", ""),
                              cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Hardware", ""),
                              _hardwareContents);

            if (loadScene())
                _showSceneCheckbox->setText("Show Scene");
            else
            {
                std::cerr << "Error: loadScene() failed." << std::endl;
                _showSceneCheckbox->setValue(false);
                return;
            }

        }

        if (_showSceneCheckbox->getValue())
            cvr::PluginHelper::getObjectsRoot()->addChild(_box->transform);
        else
            cvr::PluginHelper::getObjectsRoot()->removeChild(_box->transform);
    }
    else if (item == _xrayViewCheckbox)
    {
        bool transparent = _xrayViewCheckbox->getValue();
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
    else if (item == _displayComponentTexturesCheckbox)
    {
        Component::_displayTexturesUni->setElement(0,_displayComponentTexturesCheckbox->getValue());
        Component::_displayTexturesUni->dirty();
    }
    else if (item == _loadPowerButton)
    {
        std::string selectedNames = "";

        if (_selectionModeCheckbox->getValue())
        {
            int selections = 0;
            std::set<Component *>::iterator sit;
            for (sit = _components.begin(); sit != _components.end(); sit++)
            {
                if ((*sit)->selected)
                {
                    if (selectedNames == "")
                        selectedNames = "&name=";
                    else
                        selectedNames += ",";
                    selectedNames += (*sit)->name;
                    selections++;
                }
            }
            if (_components.size() == selections) // we grabbed all of them
                selectedNames = "";
            else if (selections == 0) // shouldn't poll anything
                selectedNames = "&name=null";

            size_t pos;
            while ((pos = selectedNames.find(' ')) != std::string::npos)
            {
                selectedNames.replace(pos,1,"%20");
            }
        }

        utl::downloadFile(cvr::ConfigManager::getEntry("download", "Plugin.GreenLight.Power", "")+selectedNames,
                          cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Power", ""),
                          _powerContents);

        if (!_displayPowerCheckbox)
        {
            std::ifstream file;
            file.open(cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "").c_str());
            if (file)
            {
                _displayPowerCheckbox = new cvr::MenuCheckbox("Display Power Consumption",false);
                _displayPowerCheckbox->setCallback(this);
                _powerMenu->addItem(_displayPowerCheckbox);
            }
            file.close();
        }

        if (!_legendText)
        {
            _legendText = new cvr::MenuText("Low    <--Legend-->    High");
            _powerMenu->addItem(_legendText);
        }

        if (!_legendGradient)
        {
            osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
            tex->setInternalFormat(GL_RGBA32F_ARB);
            tex->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
            tex->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
            tex->setResizeNonPowerOfTwoHint(false);  

            osg::ref_ptr<osg::Image> data = new osg::Image;
            data->allocateImage(100, 1, 1, GL_RGBA, GL_FLOAT);  

            for (int i = 0; i < 100; i++)
            {
                osg::Vec3 color = wattColor(i+1,1,101);
                for (int j = 0; j < 3; j++)
                {
                    ((float *)data->data(i))[j] = color[j];
                }
                ((float *)data->data(i))[3] = 1;
            }

            data->dirty();
            tex->setImage(data.get());

            _legendGradient = new cvr::MenuImage(tex,450,50);
            _powerMenu->addItem(_legendGradient);
        }

        if (!_legendTextOutOfRange)
        {
            _legendTextOutOfRange = new cvr::MenuText(" Off     | Too Low  | Too High");
            _powerMenu->addItem(_legendTextOutOfRange);
        }

        if (!_legendGradientOutOfRange)
        {
            osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
            tex->setInternalFormat(GL_RGBA32F_ARB);
            tex->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
            tex->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
            tex->setResizeNonPowerOfTwoHint(false);  

            osg::ref_ptr<osg::Image> data = new osg::Image;
            data->allocateImage(3, 1, 1, GL_RGBA, GL_FLOAT);  

            for (int i = 0; i < 3; i++)
            {
                osg::Vec3 color = wattColor(i*2,3,3);
                for (int j = 0; j < 3; j++)
                {
                    ((float *)data->data(i))[j] = color[j];
                }
                ((float *)data->data(i))[3] = 1;
            }

            data->dirty();
            tex->setImage(data.get());

            _legendGradientOutOfRange = new cvr::MenuImage(tex,450,50);
            _powerMenu->addItem(_legendGradientOutOfRange);
        }

        if (_displayPowerCheckbox->getValue())
        {
            setPowerColors(true);
        }
    }
    else if (item == _displayPowerCheckbox)
    {
        setPowerColors(_displayPowerCheckbox->getValue());

    }
    else if (item == _selectionModeCheckbox)
    {
        // Toggle the non-selected hardware transparencies
        Entity * ent;
        std::set< Component * >::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
        {
            if (!(*sit)->selected)
                (*sit)->setTransparency(_selectionModeCheckbox->getValue());
        }

        if (_selectionModeCheckbox->getValue())
        {
            if (_selectClusterMenu)
                _hardwareSelectionMenu->addItem(_selectClusterMenu);
            _hardwareSelectionMenu->addItem(_selectAllButton);
            _hardwareSelectionMenu->addItem(_deselectAllButton);
        }
        else
        {
            if (_selectClusterMenu)
                _hardwareSelectionMenu->removeItem(_selectClusterMenu);
            _hardwareSelectionMenu->removeItem(_selectAllButton);
            _hardwareSelectionMenu->removeItem(_deselectAllButton);
        }
    }
    else if (item == _selectAllButton || item == _deselectAllButton)
    {
        std::set< Component * >::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
            selectComponent(*sit, item == _selectAllButton);
    }
    else if ((chit = _clusterCheckbox.find(dynamic_cast<cvr::MenuCheckbox *>(item))) != _clusterCheckbox.end())
    {
        cvr::MenuCheckbox * checkbox = *chit;

        std::map< std::string, std::set< Component * > * >::iterator cit = _cluster.find(checkbox->getText());
        if (cit == _cluster.end())
        {
            std::cerr << "Error: Cluster checkbox selected without a matching cluster (" << checkbox->getText() << ")" << std::endl;
            checkbox->setValue(checkbox->getValue());
            return;
        }

        std::set< Component * > * cluster = cit->second;
        selectCluster(cluster, checkbox->getValue());
    }
}

void GreenLight::preFrame()
{
    // update mouse and wand intersection with components
    if (_box)
    {
        // continue animations
        for (int d = 0; d < _door.size(); d++)
            _door[d]->handleAnimation();
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->handleAnimation();

        if (cvr::ComController::instance()->isMaster())
            handleHoverOver(cvr::PluginHelper::getMouseMat(), _mouseOver);
        else
            handleHoverOver(cvr::PluginHelper::getHandMat(), _wandOver);
    }
}

void GreenLight::postFrame()
{
}

bool GreenLight::keyEvent(bool keyDown, int key, int mod)
{
    return false;
}

bool GreenLight::buttonEvent(int type, int button, int hand, const osg::Matrix& mat)
{
    if (type != cvr::BUTTON_DOWN || button != 0)
        return false;

    if (!_box)
        return false;

    // Should be hovering over it
    if (_wandOver)
    {
        Component * comp = _wandOver->asComponent();
        if (comp)
        {
            selectComponent( comp, !comp->selected );
        }
        else // _wandOver is a rack/door/etc.
        {
            _wandOver->beginAnimation();

            // Handle group animations
            std::list<Entity *>::iterator eit;
            for (eit = _wandOver->group.begin(); eit != _wandOver->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }
        }

        return true;
    }

    return false;
}

bool GreenLight::mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat)
{
    // Left Button Click
    if (type != cvr::MOUSE_BUTTON_DOWN || button != 0)
        return false;

    if (!_box)
        return false;

    // Should be hovering over it
    if (_mouseOver)
    {
        Component * comp = _mouseOver->asComponent();
        if (comp)
        {
            selectComponent( comp, !comp->selected );
        }
        else // _mouseOver is a rack/door/etc.
        {
            _mouseOver->beginAnimation();

            // Handle group animations
            std::list<Entity *>::iterator eit;
            for (eit = _mouseOver->group.begin(); eit != _mouseOver->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }
        }

        return true;
    }

    return false;
}
