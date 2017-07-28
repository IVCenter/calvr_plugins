#include "Catalyst.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/ComController.h>
#include <PluginMessageType.h>

#include <sstream>
#include <iostream>
#include <fstream>

#include <osgDB/ReadFile>
#include <osgText/Text>
#include <osg/Geometry>
#include <osg/StateAttribute>
#include <osg/LineWidth>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/PositionAttitudeTransform>

#include <osgDB/FileUtils>

using namespace cvr;

CVRPLUGIN(Catalyst)

Catalyst::Catalyst()
{
    // set font to use
    _font = osgText::readFontFile("/home/calvr/CalVR/resources/arial.ttf");
    _fontSize = ConfigManager::getFloat("Plugin.Catalyst.FontSize", 20.0);
    _textColor.set(1.0, 1.0, 1.0, 1.0);
    _frameColor.set(0.3, 0.3, 0.3, 1.0);
}

// will read data on master and then distribute
void Catalyst::parseMetaData(std::string filename, std::map<cvr::MenuItem* , PanoMetadata>& data, cvr::SubMenu *subMenu)
{

    int length = 0;
    std::stringstream ss;

    // only read file on master // send json string to slaves
    if( cvr::ComController::instance()->isMaster() )
    {
	std::ifstream ifs(filename.c_str());

	ss << ifs.rdbuf();
   
	length = ss.str().size(); 
	char jsonArray[length];
	ss.read(jsonArray, length);

	// read add buffer
	ss.str(std::string());
	ss << jsonArray;

	// send length and then array
	ComController::instance()->sendSlaves(&length, sizeof(length));
	ComController::instance()->sendSlaves(jsonArray, length * sizeof(char));
    }
    else
    {
	ComController::instance()->readMaster(&length, sizeof(length));
	char jsonArray[length];
	ComController::instance()->readMaster(jsonArray, length * sizeof(char));
	ss << jsonArray;
    }
    
	
    // parse json string stream	 
    Json::Reader reader;
    Json::Value jsonRoot;
  
    // parse mounted json data 
    if( !reader.parse(ss, jsonRoot,false) )
    {
	std::cerr << "Error parsing json\n";
	return;
    }
    
    Json::Value & currentArray = jsonRoot["cavecams"]["entries"];
    
    // read data and create menu items 
    cvr::MenuButton* tempButton = NULL;

    for (int i = 0; i < currentArray.size();i++)
    {
        // create button
        tempButton = new cvr::MenuButton(currentArray[i]["name"].asString());
        tempButton->setCallback(this);
        subMenu->addItem(tempButton);

        // add meta data to map
        data[tempButton].title = currentArray[i]["name"].asString();
        data[tempButton].description = currentArray[i]["description"].asString();
        data[tempButton].leftImage = currentArray[i]["panoproc_left"].asString();
        data[tempButton].rightImage = currentArray[i]["panoproc_right"].asString();
    }
}

Catalyst::~Catalyst()
{
}

// little icon use
osg::Geode* Catalyst::createIcon(osg::Vec3 position, osg::Texture2D* texture, bool enableBlending)
{
    osg::Geode* geode = new osg::Geode();
    geode->addDrawable(osg::createTexturedQuadGeometry(osg::Vec3(-(texture->getImage()->s() * 0.5) + position.x(), position.y(), 
                                                                 -(texture->getImage()->t() * 0.5) + position.z()),
                                                       osg::Vec3(texture->getImage()->s(),0.0f,0.0), osg::Vec3(0.0f,0.0f, texture->getImage()->t())));
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    
    if( enableBlending )
    {
        geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
        geode->getOrCreateStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );
    }
    return geode;
}

// dont init from the start create on the fly destroy and replace when needed TODO
void Catalyst::createDescriptionPanel( osg::PositionAttitudeTransform* transform, PanoMetadata& data, osg::Vec3 pos, osg::Vec4 textColor, osg::Vec4 frameColor, float fontSize)
{
    // remove all child before building new node
    while( transform->getNumChildren() )
        transform->removeChild(transform->getChild(0));

    transform->setPosition(pos);
    transform->setScale(osg::Vec3(0.5, 0.5, 0.5));
    
    // max width determine by logo attached to all screens
    float width = _logoHeaderTexture->getImage()->s();

    // load header 
    osg::Geode* headerGeode = createIcon(osg::Vec3(0.0, 0.0, -(_logoHeaderTexture->getImage()->t() * 0.5)), _logoHeaderTexture);
    transform->addChild(headerGeode);

    osg::BoundingBox headerBound = headerGeode->getBoundingBox();

    osg::Geode* geode = new osg::Geode(); 

    // add title
    osgText::Text* titleText = new osgText::Text;
    titleText->setUseVertexBufferObjects(true);
    titleText->setFont(_font);
    titleText->setColor(textColor);
    titleText->setCharacterSize(fontSize * 2);
    titleText->setPosition(osg::Vec3(-(width * 0.45), 0.0, headerBound.zMin() - fontSize));
    titleText->setAxisAlignment(osgText::Text::XZ_PLANE);
    titleText->setAlignment(osgText::Text::LEFT_TOP);
    titleText->setFontResolution(40,40);
    titleText->setLineSpacing(0.25); // set line spacing (is a percentage of font size) 
    titleText->setText(data.title); // for spacing will just use newline characters
    geode->addDrawable(titleText);

    osg::BoundingBox bound = geode->getBoundingBox();

    // add description
    osgText::Text* desText = new osgText::Text;
    desText->setUseVertexBufferObjects(true);
    desText->setFont(_font);
    desText->setColor(textColor);
    desText->setCharacterSize(fontSize);
    desText->setPosition(osg::Vec3( -(width * 0.45), 0.0, bound.zMin() - fontSize)); // todo shift in x a percentage
    desText->setAxisAlignment(osgText::Text::XZ_PLANE);
    desText->setAlignment(osgText::Text::LEFT_TOP);
    desText->setFontResolution(40,40);
    desText->setLineSpacing(0.4); // set line spacing (is a percentage of font size)
    desText->setMaximumWidth( width * 0.9);
    // TODO need to drop off the spacing so I can get the actual bound of the text so I can create a frame to indicate active

    //desText->setText(_panoMetaData->at(_currentPanoSetIndex)->panos.at(_currentPanoIndex).description); // for spacing will just use newline characters
    desText->setText(data.description); // for spacing will just use newline characters
    geode->addDrawable(desText);
    
    // add footer
    bound = geode->getBoundingBox();

    // load footer 
    osg::Geode* footerGeode = createIcon(osg::Vec3(0.0, 0.0, bound.zMin() - fontSize -(_logoFooterTexture->getImage()->t() * 0.5)), _logoFooterTexture); 
    transform->addChild(footerGeode);

    transform->addChild(geode);
    
    //compute bound of all text, create frame based on that
    osg::BoundingBox footerBound = footerGeode->getBoundingBox();
    
    // create frame
    osg::Geometry* frameGeom = new osg::Geometry();
    osg::Vec3Array* vertices = new osg::Vec3Array();
    osg::Vec4Array* colors = new osg::Vec4Array();

    // expand bound to include header (cheap hack :) )
    footerBound.expandBy(osg::Vec3(footerBound.corner(0).x(), 0.2, 0.0));
    
    //std::cerr << "Bound corner 2: " << bound.corner(2).x() << " " << bound.corner(2).y() << " " << bound.corner(2).z() << std::endl;
    //std::cerr << "Bound corner 3: " << bound.corner(3).x() << " " << bound.corner(3).y() << " " << bound.corner(3).z() << std::endl;
    //std::cerr << "Bound corner 7: " << bound.corner(7).x() << " " << bound.corner(7).y() << " " << bound.corner(7).z() << std::endl;
    //std::cerr << "Bound corner 6: " << bound.corner(6).x() << " " << bound.corner(6).y() << " " << bound.corner(6).z() << std::endl;
    
    vertices->push_back(footerBound.corner(2));
    vertices->push_back(footerBound.corner(6));
    vertices->push_back(footerBound.corner(7));
    vertices->push_back(footerBound.corner(3));
    colors->push_back(frameColor);

    frameGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0 ,vertices->size()));
    frameGeom->setVertexArray(vertices);
    //frameGeom->setColorArray(colors, osg::Array::BIND_OVERALL);
    
    //geode->addDrawable(frameGeom);
    osg::Geode* frameGeode = new osg::Geode();
    osg::Material* material = new osg::Material();
    material->setDiffuse(osg::Material::FRONT_AND_BACK, frameColor);
    frameGeode->getOrCreateStateSet()->setAttribute(material);
    frameGeode->addDrawable(frameGeom);
    transform->addChild(frameGeode);

    // update bound of menu
    osg::BoundingBox total;
    total.expandBy(headerBound);
    total.expandBy(footerBound);
    _descriptionMenu->setBoundingBox(total);
}

// TODO add to background and display a loader, and keep it interactive
// once all nodes can see the file exit check
bool Catalyst::checkCache(std::string fileName)
{
    bool fileCheck = true;

    // just do check on master
    if( cvr::ComController::instance()->isMaster() )
    {	
	if( osgDB::findFileInDirectory(fileName, _cacheDirectory, osgDB::CASE_SENSITIVE).compare("") == 0 )
	{
	    //copy data to cache
	    fileCheck = osgDB::copyFile(_remoteMount + fileName, _cacheDirectory + fileName);
	}

	ComController::instance()->sendSlaves(&fileCheck, sizeof(fileCheck));

    }
    else // make slaves wait
    {
	ComController::instance()->readMaster(&fileCheck, sizeof(fileCheck));
    }
    return fileCheck;
}

void Catalyst::menuCallback(cvr::MenuItem * item)
{

    // access menu option that was selected and load image
    if( _catalystFileMap.find(item) !=  _catalystFileMap.end() )
    {
        // remove description panel and remove pan if exists
        PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD, NULL);
        
	// check for file in cache 
	checkCache(_catalystFileMap[item].leftImage);	
	checkCache(_catalystFileMap[item].rightImage);	
	
        // pano and metadata exists load pano
        createDescriptionPanel(_description, _catalystFileMap[item], _descriptionPos, _textColor, _frameColor,_fontSize);

        // need to send a request object
        PanLoadRequest plr;
        plr.name = _catalystFileMap[item].title;
        plr.leftFile = _catalystFileMap[item].leftImage;
        plr.rightFile = _catalystFileMap[item].rightImage;
        plr.rotationOffset = 0.0;
        plr.plugin = "Catalyst";
        plr.enableRemoval = false;
        plr.pluginMessageType = CAT_PAN_UNLOADED; // will get message once pan is unloaded
        plr.loaded = false;

        // fade selection screen and fade in Pano Description and fade in first pano from set // TODO
        PluginHelper::sendMessageByName("PanoViewLOD",PAN_LOAD_REQUEST,(char*)&plr); 

    }
    else if( item == _descriptionToggle )
    {
        if( _descriptionToggle->getValue() )
        {
            // make visible
            _description->setNodeMask(~0);
        }
        else
        {
            // hide
            _description->setNodeMask(0);
        }
    }
    else if( item == _removeButton)
    {
        // remove description panel and remove pan
        PluginHelper::sendMessageByName("PanoViewLOD",PAN_UNLOAD, NULL);
      
        remove(); 
    }

}

void Catalyst::remove()
{
    // received unload message clean up 
    while( _description->getNumChildren() )
        _description->removeChild(_description->getChild(0));

}

void Catalyst::message(int type, char *&data, bool collaborative)
{
    if( type == CAT_PAN_UNLOADED)
    {
        remove();
    }
}

bool Catalyst::init()
{

    std::cerr << "Initializing Catalyst\n";
    
    // create default menu elements
    _catalystMenu = new cvr::SubMenu("Catalyst", "Catalyst");
    _catalystMenu->setCallback(this);
    
    _catalystloadMenu = new cvr::SubMenu("Load", "Load");
    _catalystMenu->addItem(_catalystloadMenu);
   
    _descriptionToggle = new cvr::MenuCheckbox("Description", true);
    _descriptionToggle->setCallback(this); 
    _catalystMenu->addItem(_descriptionToggle);
    
    _removeButton = new  cvr::MenuButton("Remove");
    _removeButton->setCallback(this); 
    _catalystMenu->addItem(_removeButton);

    _cacheDirectory = ConfigManager::getEntry("Plugin.Catalyst.CacheDirectory");
    _remoteMount = ConfigManager::getEntry("Plugin.Catalyst.RemoteMount");

    // hold pano meta data
    //_panoMetaData = new std::vector< PanoSetMetadata* >();

    std::string metaFileName = ConfigManager::getEntry("Plugin.Catalyst.ConfigJSON");
    std::string textureDir = ConfigManager::getEntry("Plugin.Catalyst.Textures");

    // parse json data file
    parseMetaData(metaFileName, _catalystFileMap, _catalystloadMenu);

    MenuSystem::instance()->addMenuItem(_catalystMenu);

    // create Description Node
    _description = new osg::PositionAttitudeTransform();
    
    _descriptionMenu = new SceneObject("MetaData",false,true,false,false,false);
    _descriptionMenu->setBoundsCalcMode(SceneObject::MANUAL);
    PluginHelper::registerSceneObject(_descriptionMenu,"Catalyst");
    _descriptionMenu->setPosition(osg::Vec3(ConfigManager::getVec3("Plugin.Catalyst.DescriptionPosition")));
    _descriptionMenu->addChild(_description);
    _descriptionMenu->attachToScene();
    _descriptionMenu->setNavigationOn(false);

    // load in default logo texture header
    _logoHeaderTexture = new osg::Texture2D;
    _logoHeaderTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
    _logoHeaderTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
    _logoHeaderTexture->setWrap(osg::Texture::WRAP_R,osg::Texture::REPEAT);
    _logoHeaderTexture->setResizeNonPowerOfTwoHint(false);
    _logoHeaderTexture->setImage(osgDB::readImageFile(textureDir + "/TopBanner.jpg"));
   
    //std::cerr << "Header Width and Height: " << _logoHeaderTexture->getImage()->s() << " " << _logoHeaderTexture->getImage()->t() << std::endl;
    
    // load logo texture footer
    _logoFooterTexture = new osg::Texture2D;
    _logoFooterTexture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
    _logoFooterTexture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
    _logoFooterTexture->setWrap(osg::Texture::WRAP_R,osg::Texture::REPEAT);
    _logoFooterTexture->setResizeNonPowerOfTwoHint(false);
    _logoFooterTexture->setImage(osgDB::readImageFile(textureDir + "/BottomBanner.jpg"));
    
    //std::cerr << "Footer Width and Height: " << _logoFooterTexture->getImage()->s() << " " << _logoFooterTexture->getImage()->t() << std::endl;

    return true;
}

