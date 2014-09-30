#include "ImageSequence.h"
#include "CVRImageSequence.h"

#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
//#include <osg/ImageSequence>
#include <osg/Texture2D>

using namespace cvr;

CVRPLUGIN(ImageSequence)

ImageSequence::ImageSequence()
{
    _activeObject = NULL;
}

ImageSequence::~ImageSequence()
{
}

bool ImageSequence::init()
{
    _isMenu = new SubMenu("ImageSequence");
    PluginHelper::addRootMenuItem(_isMenu);

    _loadMenu = new SubMenu("Load");
    _isMenu->addItem(_loadMenu);

    std::vector<std::string> tags;
    ConfigManager::getChildren("Plugin.ImageSequence.Files",tags);

    for(int i = 0; i < tags.size(); ++i)
    {
	MenuButton * button = new MenuButton(tags[i]);
	button->setCallback(this);
	_loadMenu->addItem(button);
	_loadButtons.push_back(button);

	SequenceSet * ss = new SequenceSet;
	ss->path = ConfigManager::getEntry("path",std::string("Plugin.ImageSequence.Files.") + tags[i],"");
	ss->start = ConfigManager::getInt("start",std::string("Plugin.ImageSequence.Files.") + tags[i],0);
	ss->frames = ConfigManager::getInt("frames",std::string("Plugin.ImageSequence.Files.") + tags[i],0);
	_sets.push_back(ss);
    }

    int index = ConfigManager::getInt("value","Plugin.ImageSequence.AutoStart",-1);
    if(index >=0 && index < _sets.size())
    {
	_autoStart = true;
	_autoStartIndex = index;
    }
    else
    {
	_autoStart = false;
    }

    _removeButton = new MenuButton("Remove");
    _removeButton->setCallback(this);
    _isMenu->addItem(_removeButton);

    return true;
}

void ImageSequence::preFrame()
{
    if(_autoStart)
    {
	menuCallback(_loadButtons[_autoStartIndex]);
	_autoStart = false;
    }
}

void ImageSequence::menuCallback(MenuItem * item)
{

    if(item == _removeButton)
    {
	if(_activeObject)
	{
	    delete _activeObject;
	    _activeObject = NULL;
	}
	return;
    }

    for(int i = 0; i < _loadButtons.size(); ++i)
    {
	if(item == _loadButtons[i])
	{
	    if(_activeObject)
	    {
		menuCallback(_removeButton);
	    }

	    osg::Geode* geode = new osg::Geode;
	    geode->addDrawable(osg::createTexturedQuadGeometry(osg::Vec3(-0.5f,0.0f,-0.5f), osg::Vec3(1.0f,0.0f,0.0), osg::Vec3(0.0f,0.0f,1.0f)));

	    osg::Matrix m;
	    m.makeScale(osg::Vec3(4000,1,4000));

	    osg::MatrixTransform * mt = new osg::MatrixTransform();
	    mt->setMatrix(m);
	    mt->addChild(geode);

	    _activeObject = new SceneObject("Image Sequence",false,true,false,true,true);
	    _activeObject->addChild(mt);
	    PluginHelper::registerSceneObject(_activeObject,"ImageSequence");
	    _activeObject->attachToScene();

	    osg::ref_ptr<CVRImageSequence> imageSequence = new CVRImageSequence;
	    imageSequence->setMode(osg::ImageSequence::LOAD_AND_DISCARD_IN_UPDATE_TRAVERSAL);


	    for(int j = 0; j < _sets[i]->frames; ++j)
	    {
		char buffer[2048];

		snprintf(buffer,2048,_sets[i]->path.c_str(),(_sets[i]->start+j));

		imageSequence->addImageFile(buffer);
	    }

	    imageSequence->setLength(90);
	    imageSequence->play();

	    osg::Texture2D* texture = new osg::Texture2D;
	    texture->setFilter(osg::Texture::MIN_FILTER,osg::Texture::LINEAR);
	    texture->setFilter(osg::Texture::MAG_FILTER,osg::Texture::LINEAR);
	    texture->setWrap(osg::Texture::WRAP_R,osg::Texture::REPEAT);
	    texture->setResizeNonPowerOfTwoHint(false);
	    texture->setImage(imageSequence.get());

	    osg::StateSet * stateset = geode->getOrCreateStateSet();
	    stateset->setTextureAttributeAndModes(0,texture,osg::StateAttribute::ON);
	    break;
	}
    }

}
