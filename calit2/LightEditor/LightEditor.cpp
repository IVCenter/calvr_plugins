//Std headers:
#include <string> 
#include <stdio.h>

// CalVR Headers:
#include <kernel/ComController.h>
#include <kernel/PluginHelper.h>

// OSG headers:
#include <osgDB/ReadFile>
#include <osg/Texture2D>
#include <osg/Light>
#include <osg/LightSource> 
#include <osgUtil/IntersectVisitor>

// Local plugin headers:
#include "LightEditor.h"
#include "LightLoader.h"

CVRPLUGIN(LightEditor)

void initScene() // Temp function to create a visual scene we can use - delete later
{
    //////////////////MODEL LOADING//////////////////
    const std::string modelFile = cvr::ConfigManager::getEntry("value","Plugin.LightEditor.Scene","");
    osg::ref_ptr<osg::Node> modelNode = osgDB::readNodeFile(modelFile);
    if (!modelNode.valid())
    {
        std::cerr << "Couldn't load model from " << modelFile << std::endl;
        return;
    }    

    // Add node -> transform -> root
    osg::ref_ptr<osg::MatrixTransform> trans = new osg::MatrixTransform();
    trans->addChild(modelNode.get());
    trans->setMatrix(osg::Matrix::identity());
    cvr::PluginHelper::getObjectsRoot()->addChild(trans.get());
}

LightEditor::LightEditor()
{  
    std::cerr << "LightEditor created." << std::endl;
}

void LightEditor::menuCallback(cvr::MenuItem * item)
{
    // Used later if necessary 
    bool isDir, isPoi, isSpo, isAmb, isDif, isSpe, isCon, isLin, isQua, isExp, isCut;

    if (item == _createNewLightButton)
    {
        addNewLight();

        osg::Vec4 pos = osg::Vec4(0,500.0,0,1);

        if (cvr::ComController::instance()->isMaster())
            pos = pos * cvr::PluginHelper::getMouseMat();
        else
            pos = pos * cvr::PluginHelper::getHandMat();

        mLightManager->PhysicalPosition(pos * cvr::PluginHelper::getWorldToObjectTransform());

        // Let's also default to point lights
        mLightManager->LightType(LightManager::POINT);

    }

    else if (item == _selectLightList)
    {
        if (mLightManager->selectLightByName(_selectLightList->getValue()))
                updateEditLightMenu();
    }

    else if (item == _graphicModelsCheckbox)
    {        
        if (_graphicModelsCheckbox->getValue())
            mLightManager->enableGraphicModels();
        else
            mLightManager->disableGraphicModels();
    }

    else if (item == _saveLightsButton)
    {
        static std::string lightXmlFile = cvr::ConfigManager::getEntry("value","Plugin.LightEditor.LightXmlFile","");

        if (lightXmlFile.empty())
            std::cerr << "Attention: No XML filepath given in config file. Cannot save lights until a filepath exists." << std::endl;
        else
            LightLoader::saveLights(lightXmlFile.c_str(), mLightManager);
    }

    // Color  for Ambient, Diffuse, and Specular
    else if (item == _elR || item == _elG || item == _elB)
    {
        osg::Vec4 (LightManager::*getFunc)();
        void (LightManager::*setFunc)(const osg::Vec4);

        std::string colorType = _elColorTypeList->getValue();
        if (colorType == "Ambient")
        {
            getFunc = &LightManager::Ambient;
            setFunc = &LightManager::Ambient;
        }
        else if (colorType == "Diffuse")
        {
            getFunc = &LightManager::Diffuse;
            setFunc = &LightManager::Diffuse;
        }
        else // Specular
        {
            getFunc = &LightManager::Specular;
            setFunc = &LightManager::Specular;
        }
    
        if (item == _elR)
        {
             osg::Vec4 val = (mLightManager->*getFunc)();
             val.set(_elR->getValue(), val.y(), val.z(), val.w());
             (mLightManager->*setFunc)(val);
        }
        if (item == _elG)
        {
             osg::Vec4 val = (mLightManager->*getFunc)();
             val.set(val.x(), _elG->getValue(), val.z(), val.w());
             (mLightManager->*setFunc)(val);
        }
        if (item == _elB)
        {
             osg::Vec4 val = (mLightManager->*getFunc)();
             val.set(val.x(), val.y(), _elB->getValue(), val.w());
             (mLightManager->*setFunc)(val);
        }
    }

    // Factor  for attenuation
    else if (item == _elAttenuation)
    {
        void (LightManager::*setFunc)(const float);

        std::string attenuation = _elAttenuationList->getValue();
        if (attenuation == "Constant")
        {
            setFunc = &LightManager::ConstantAttenuation;
        }
        else if (attenuation == "Linear")
        {
            setFunc = &LightManager::LinearAttenuation;
        }
        else // Quadratic
        {
            setFunc = &LightManager::QuadraticAttenuation;
        }

        (mLightManager->*setFunc)(_elAttenuation->getValue());        
    }    


    //  for spot settings
    else if (item == _elSpotExponent)
    {
        mLightManager->SpotExponent(_elSpotExponent->getValue());
    }

    else if (item == _elSpotCutoff)
    {
        mLightManager->SpotCutoff(_elSpotCutoff->getValue());
    }

    // Treat Directional, Point, and Spot buttons as Radio Buttons
    else if (item == _elLightTypeList)
    {
        std::string lightType = _elLightTypeList->getValue();
        LightManager::Type type;
        if (lightType == "Directional")
            type = LightManager::DIRECTIONAL;
        else if (lightType == "Spot")
            type = LightManager::SPOT;
        else // "Point"
            type = LightManager::POINT;

        mLightManager->LightType(type);
    } 

    if (_elToggleEnable == item)
    {
        if (_elToggleEnable->getValue())
            mLightManager->enableLight();
        else
            mLightManager->disableLight();
    }

    // Make sure to update the menu
    updateEditLightMenu();
}

bool LightEditor::mouseButtonEvent (int type, int button, int x, int y, const osg::Matrix &mat)
{
    switch(type)
    {
        case cvr::MOUSE_BUTTON_DOWN:
            return buttonEvent(cvr::BUTTON_DOWN, button, 0, mat);
        case cvr::MOUSE_BUTTON_UP:
            return buttonEvent(cvr::BUTTON_UP, button, 0, mat);
        case cvr::MOUSE_DRAG:
            return buttonEvent(cvr::BUTTON_DRAG, button, 0, mat);
        case cvr::MOUSE_DOUBLE_CLICK:
            return buttonEvent(cvr::BUTTON_DOUBLE_CLICK, button, 0, mat);
        default:
            return false;            
    }
}


bool LightEditor::buttonEvent(int type, int button, int hand, const osg::Matrix& mat)
{
    if (button != 0)
        return false;

    static osg::Matrix invLastWand = osg::Matrix();
    static osg::Vec4 * pos = NULL;
    static osg::Vec3 * dir = NULL;

    if (type == cvr::BUTTON_DOWN || type == cvr::BUTTON_DOUBLE_CLICK)
    {
        osg::Vec3 pointerStart, pointerEnd;
        std::vector<IsectInfo> isecvec;

        pointerStart = mat.getTrans();
        pointerEnd.set(0.0f, 10000.0f, 0.0f);
        pointerEnd = pointerEnd * mat;

        isecvec = getObjectIntersection(cvr::PluginHelper::getScene(),
            pointerStart, pointerEnd);

        // If we didn't intersect, get out of here
        if (isecvec.size() == 0 || !mLightManager->selectLightByGeodePtr(isecvec[0].geode))
            return false;

        invLastWand = osg::Matrix::inverse(mat * cvr::PluginHelper::getWorldToObjectTransform());

        if (type == cvr::BUTTON_DOUBLE_CLICK && mLightManager->LightType() == LightManager::SPOT)
            dir = new osg::Vec3(mLightManager->SpotDirection());
        else
            pos = new osg::Vec4(mLightManager->PhysicalPosition());

        updateEditLightMenu();

        return true;
    }
    else if (type == cvr::BUTTON_DRAG || type == cvr::BUTTON_UP)
    {
        if (!pos && !dir) // We can't do anything
        {
            return false;
        }
        else if (pos) // Changing Physical Light Position
        {
            mLightManager->PhysicalPosition(*pos * invLastWand * mat * 
                        cvr::PluginHelper::getWorldToObjectTransform());
        }
        else // Changing Spot Direction
        {
            osg::Quat rotQ = (invLastWand * mat * cvr::PluginHelper::getWorldToObjectTransform()).getRotate();
            mLightManager->SpotDirection(*dir * osg::Matrix(rotQ));
        }

        if (type == cvr::BUTTON_UP)
        {
            pos = NULL;
            dir = NULL;
        }

        updateEditLightMenu();

        return true;
    }

    return false;
}

bool LightEditor::init()
{
    // Menu Setup
    _lightMenu = new cvr::SubMenu("LightEditor","LightEditor");
    _lightMenu->setCallback(this);
    cvr::PluginHelper::addRootMenuItem(_lightMenu);

    _createNewLightButton = new cvr::MenuButton("Create New Light");
    _createNewLightButton->setCallback(this);
    _lightMenu->addItem(_createNewLightButton);

    _selectedLightText = new cvr::MenuText("Selected Light:");
    _selectedLightText->setCallback(this);
    _lightMenu->addItem(_selectedLightText);

    _selectLightList = new cvr::MenuList;
    _selectLightList->setCallback(this);
    _lightMenu->addItem(_selectLightList);

    initEditLightMenu();

    bool graphicModels = cvr::ConfigManager::getBool("GraphicModels","Plugin.LightEditor",true);

    _graphicModelsCheckbox = new cvr::MenuCheckbox("Graphic Models",graphicModels);
    _graphicModelsCheckbox->setCallback(this);
    _lightMenu->addItem(_graphicModelsCheckbox);

    _saveLightsButton = new cvr::MenuButton("Save Lights");
    _saveLightsButton->setCallback(this);
    _lightMenu->addItem(_saveLightsButton);
    // End Menu Setup
    
    initScene();	// Temp function to create a visual scene we can use

    // Setup the LightManager
    mLightManager = new LightManager();

    if (graphicModels)
        mLightManager->enableGraphicModels();
    else
        mLightManager->disableGraphicModels();

    static std::string lightXmlFile = cvr::ConfigManager::getEntry("value","Plugin.LightEditor.LightXmlFile","");
    // If lightsXML is not an empty std::string, we need to load the lights found at the file given.
    if (!lightXmlFile.empty())
        LightLoader::loadLights(lightXmlFile.c_str(), mLightManager);
    repopulateSelectLightMenu();

    // Setup the lighting shader  
    mLightShading = new LightShading(mLightManager);

    return true;
}

// initializes the Edit Light Menu
void LightEditor::initEditLightMenu()
{
    _elPopup = new cvr::PopupMenu("Edit Light");
    _elPopup->setVisible(false);

    _elToggleEnable = new cvr::MenuCheckbox("Enabled", true);
    _elToggleEnable->setCallback(this);
    _elPopup->addMenuItem(_elToggleEnable);

    _elLightTypeText = new cvr::MenuText("Light Type:");
    _elPopup->addMenuItem(_elLightTypeText);

    std::vector<std::string> lightTypes;
    lightTypes.push_back("Directional");
    lightTypes.push_back("Point");
    lightTypes.push_back("Spot");
    _elLightTypeList = new cvr::MenuList();
    _elLightTypeList->setValues(lightTypes);
    _elLightTypeList->setCallback(this);
    _elPopup->addMenuItem(_elLightTypeList);

    _elColorTypeText = new cvr::MenuText("Editing Color Type:");
    _elPopup->addMenuItem(_elColorTypeText);

    std::vector<std::string> colorTypes;
    colorTypes.push_back("Ambient");
    colorTypes.push_back("Diffuse");
    colorTypes.push_back("Specular");
    _elColorTypeList = new cvr::MenuList();
    _elColorTypeList->setValues(colorTypes);
    _elColorTypeList->setCallback(this);
    _elPopup->addMenuItem(_elColorTypeList);


    _elR = new cvr::MenuRangeValue("R",0,1,1,.01);
    _elR->setCallback(this);
    _elPopup->addMenuItem(_elR);

    _elG = new cvr::MenuRangeValue("G",0,1,1,.01);
    _elG->setCallback(this);
    _elPopup->addMenuItem(_elG);

    _elB = new cvr::MenuRangeValue("B",0,1,1,.01);
    _elB->setCallback(this);
    _elPopup->addMenuItem(_elB);

    _elAttenuationText = new cvr::MenuText("Attenuation Factors");
    _elPopup->addMenuItem(_elAttenuationText);

    std::vector<std::string> attenuationTypes;
    attenuationTypes.push_back("Constant");
    attenuationTypes.push_back("Linear");
    attenuationTypes.push_back("Quadratic");
    _elAttenuationList = new cvr::MenuList();
    _elAttenuationList->setValues(attenuationTypes);
    _elAttenuationList->setCallback(this);
    _elPopup->addMenuItem(_elAttenuationList);

    _elAttenuation = new cvr::MenuRangeValue("Factor",0.0,
        cvr::ConfigManager::getFloat("value","Plugin.LightEditor.AttenuationMax",5.0), .001);
    _elAttenuation->setCallback(this);
    _elPopup->addMenuItem(_elAttenuation);

    _elToggleSpotDirection = new cvr::MenuCheckbox("Spot Direction", false);
    _elToggleSpotDirection->setCallback(this);

    _elLabelSpotDirection = new cvr::MenuText("[]");
    _elLabelSpotDirection->setCallback(this);

    _elSpotExponent = new cvr::MenuRangeValue("Spot Exponent:", 0.0,
        cvr::ConfigManager::getFloat("value","Plugin.LightEditor.SpotExponentMax",125.0), 0.25);
    _elSpotExponent->setCallback(this);

    _elSpotCutoff = new cvr::MenuRangeValue("Spot Cutoff:", 0.0, 90.0, 0.25);
    _elSpotCutoff->setCallback(this);
}

void LightEditor::preFrame()
{
    // Update shader
    mLightShading->UpdateUniforms();
}


void LightEditor::updateEditLightMenu()
{
    if (!mLightManager->isLightSelected())
    {
        _elPopup->setVisible(false);
        return;
    }

    // update enable/disable button
    bool enabled = mLightManager->LightOn();
    _elToggleEnable->setValue(enabled);

    // get the color for the pressed button
    osg::Vec4 color;
    std::string colorType = _elColorTypeList->getValue();
    if (colorType == "Ambient")
        color = mLightManager->Ambient();
    else if (colorType == "Diffuse")
        color = mLightManager->Diffuse();
    else // Specular
        color = mLightManager->Specular();

    _elR->setValue(color.x());
    _elG->setValue(color.y());
    _elB->setValue(color.z());

    _elPopup->removeMenuItem(_elToggleSpotDirection);
    _elPopup->removeMenuItem(_elLabelSpotDirection);
    _elPopup->removeMenuItem(_elSpotExponent);
    _elPopup->removeMenuItem(_elSpotCutoff);

    if (mLightManager->LightType() == LightManager::SPOT)
    {
        _elLightTypeList->matchIndexToValue("Spot");
        _elPopup->addMenuItem(_elToggleSpotDirection);
        _elPopup->addMenuItem(_elLabelSpotDirection);
        _elPopup->addMenuItem(_elSpotExponent);
        _elPopup->addMenuItem(_elSpotCutoff);
    }
    else if (mLightManager->LightType() == LightManager::DIRECTIONAL)
    {
        _elLightTypeList->matchIndexToValue("Directional");
    }
    else // POINT
    {
        _elLightTypeList->matchIndexToValue("Point");
    }

    std::string attenuationType = _elAttenuationList->getValue();
    if  (attenuationType == "Constant")
        _elAttenuation->setValue(mLightManager->ConstantAttenuation());
    else if (attenuationType == "Linear")
        _elAttenuation->setValue(mLightManager->LinearAttenuation());
    else // Quadratic
        _elAttenuation->setValue(mLightManager->QuadraticAttenuation());

    _elSpotExponent->setValue(mLightManager->SpotExponent());
    _elSpotCutoff->setValue(mLightManager->SpotCutoff());

    // update spot direction
    osg::Vec3 dir = mLightManager->SpotDirection();
    char str[40]; // Minimum value here needs to be at least (13 + 3 * precision).
    sprintf(str, "[%.2f, %.2f, %.2f]", dir.x(), dir.y(), dir.z());
    _elLabelSpotDirection->setText(str);

    _elPopup->setVisible(true);
}

void LightEditor::addNewLight()
{
    mLightManager->createNewLight();
    repopulateSelectLightMenu();
    updateEditLightMenu();
}

void LightEditor::repopulateSelectLightMenu()
{
    // grab needed names to recreate list
    std::list<std::string> names;
    mLightManager->populateLightNameList(names);

    // repopulate the list
    std::vector<std::string> listNames;
    listNames.resize(names.size(),"");

    std::string selectedName = mLightManager->isLightSelected() ? mLightManager->Name() : "";
    int selectedIndex = -1;

    std::list<std::string>::iterator j;
    int i;
    for (i=0, j = names.begin(); j != names.end(); i++,j++)
    {
        listNames[i] = *j;

        if (*j == selectedName)
            selectedIndex = i;
    }

    _selectLightList->setValues(listNames);

    if (selectedIndex != -1)
        _selectLightList->setIndex(selectedIndex);
}

// this is called if the plugin is removed at runtime
LightEditor::~LightEditor()
{
    //delete mLightManager;        

    // Menu objects
    if (_lightMenu) delete _lightMenu;
    if (_createNewLightButton) delete _createNewLightButton;
    if (_selectedLightText) delete _selectedLightText;
    if (_selectLightList) delete _selectLightList;
    if (_graphicModelsCheckbox) delete _graphicModelsCheckbox;
    if (_saveLightsButton) delete _saveLightsButton;

    // + Edit Light Menu
    if (_elPopup) delete _elPopup;

    if (_elToggleEnable) delete _elToggleEnable;

    if (_elLightTypeText) delete _elLightTypeText;
    if (_elLightTypeList) delete _elLightTypeList;

    if (_elColorTypeText) delete _elColorTypeText;
    if (_elColorTypeList) delete _elColorTypeList;

    if (_elR) delete _elR;
    if (_elG) delete _elG;
    if (_elB) delete _elB; 

    if (_elAttenuationText) delete _elAttenuationText;	
    if (_elAttenuationList) delete _elAttenuationList;
    if (_elAttenuation) delete _elAttenuation; 

    if (_elToggleSpotDirection) delete _elToggleSpotDirection;
    if (_elLabelSpotDirection) delete _elLabelSpotDirection;		
    if (_elSpotExponent) delete _elSpotExponent;
    if (_elSpotCutoff) delete _elSpotCutoff;

    std::cerr << "LightEditor destroyed." << std::endl;
}
