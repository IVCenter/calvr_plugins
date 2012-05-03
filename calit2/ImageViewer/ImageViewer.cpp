#include "ImageViewer.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>

#include <iostream>
#include <sys/stat.h>

using namespace cvr;

CVRPLUGIN(ImageViewer)

ImageViewer::ImageViewer()
{
}

ImageViewer::~ImageViewer()
{
}

bool ImageViewer::init()
{
    std::string paths = ConfigManager::getEntry("Plugin.ImageViewer.DefaultPaths");
    size_t position = 0;
    
    while(position < paths.size())
    {
	size_t lastPosition = position;
	position = paths.find_first_of(':', position);
	if(position == std::string::npos)
	{
	    size_t length = paths.size() - lastPosition;
	    if(length)
	    {
		_pathList.push_back(paths.substr(lastPosition,length));
		break;
	    }
	}
	else
	{
	    size_t length = position - lastPosition;
	    if(length)
	    {
		_pathList.push_back(paths.substr(lastPosition,length));
	    }
	    position++;
	}
    }

    _imageViewerMenu = new SubMenu("ImageViewer");
    PluginHelper::addRootMenuItem(_imageViewerMenu);

    _filesMenu = new SubMenu("Files");
    _imageViewerMenu->addItem(_filesMenu);

    _removeButton = new MenuButton("Remove All");
    _removeButton->setCallback(this);
    _imageViewerMenu->addItem(_removeButton);

    std::vector<std::string> fileNames;
    ConfigManager::getChildren("Plugin.ImageViewer.Files",fileNames);

    for(int i = 0; i < fileNames.size(); i++)
    {
	std::string tag = "Plugin.ImageViewer.Files." + fileNames[i];
	createLoadMenu(fileNames[i], tag, _filesMenu);
    }

    float x,y,z;
    x = ConfigManager::getFloat("x","Plugin.ImageViewer.GlobalOffset",0.0);
    y = ConfigManager::getFloat("y","Plugin.ImageViewer.GlobalOffset",0.0);
    z = ConfigManager::getFloat("z","Plugin.ImageViewer.GlobalOffset",0.0);
    _globalOffset = osg::Vec3(x,y,z);

    return true;
}

void ImageViewer::menuCallback(MenuItem * item)
{
    if(item == _removeButton)
    {
	for(int i = 0; i < _deleteButtons.size(); i++)
	{
	    delete _deleteButtons[i];
	}
	for(int i = 0; i < _loadedImages.size(); i++)
	{
	    PluginHelper::unregisterSceneObject(_loadedImages[i]);
	    delete _loadedImages[i];
	}
	_deleteButtons.clear();
	_loadedImages.clear();
    }

    for(int i = 0; i < _fileButtons.size(); i++)
    {
	if(item == _fileButtons[i])
	{
	    ImageObject * io = new ImageObject(_files[i]->name,false,true,false,true,false);
	    if(_files[i]->stereo)
	    {
		io->loadImages(_files[i]->fileLeft,_files[i]->fileRight);
	    }
	    else
	    {
		io->loadImages(_files[i]->fileLeft);
	    }

	    PluginHelper::registerSceneObject(io,"ImageViewer");
	    io->attachToScene();

	    io->setNavigationOn(true);

	    MenuButton * mb = new MenuButton("Delete");
	    mb->setCallback(this);
	    io->addMenuItem(mb);

	    io->setAspectRatio(_files[i]->aspectRatio);
	    io->setWidth(_files[i]->width);
	    io->setScale(_files[i]->scale);

	    osg::Matrix m;
	    m.makeRotate(_files[i]->rotation);
	    m.setTrans(_files[i]->position + _globalOffset);

	    _loadedImages.push_back(io);
	    _deleteButtons.push_back(mb);

	    return;
	}
    }

    for(int i = 0; i < _deleteButtons.size(); i++)
    {
	if(item == _deleteButtons[i])
	{
	    _loadedImages[i]->removeMenuItem(_deleteButtons[i]);
	    delete _deleteButtons[i];
	    _loadedImages[i]->detachFromScene();
	    PluginHelper::unregisterSceneObject(_loadedImages[i]);
	    delete _loadedImages[i];

	    std::vector<ImageObject*>::iterator it = _loadedImages.begin();
	    it += i;
	    _loadedImages.erase(it);

	    std::vector<MenuButton*>::iterator it2 = _deleteButtons.begin();
	    it2 += i;
	    _deleteButtons.erase(it2);

	    return;
	}
    }
}

std::string ImageViewer::findFile(std::string name)
{
    if(name.empty())
    {
	return "";
    }

    struct stat result;

    if(stat(name.c_str(), &result) == 0)
    {
	if(result.st_size)
	{
	    return name;
	}
    }

    for(int i = 0; i < _pathList.size(); i++)
    {
	std::string file = _pathList[i] + "/" + name;

	if(stat(file.c_str(), &result) == 0)
	{
	    if(result.st_size)
	    {
		return file;
	    }
	}

    }

    return "";
}

void ImageViewer::createLoadMenu(std::string tagBase, std::string tag, SubMenu * menu)
{
    std::vector<std::string> tagList;
    ConfigManager::getChildren(tag, tagList);

    if(tagList.size())
    {
	SubMenu * sm = new SubMenu(tagBase);
	menu->addItem(sm);
	for(int i = 0; i < tagList.size(); i++)
	{
	    createLoadMenu(tagList[i], tag + "." + tagList[i], sm);
	}
    }
    else
    {
	ImageInfo * ii = new ImageInfo;
	ii->name = tagBase;

	ii->fileLeft = findFile(ConfigManager::getEntry("file",tag,""));
	if(!ii->fileLeft.empty())
	{
	    ii->stereo = false;
	}
	else
	{
	    ii->fileLeft = findFile(ConfigManager::getEntry("fileLeft",tag,""));
	    ii->fileRight = findFile(ConfigManager::getEntry("fileRight",tag,""));
	    ii->stereo = true;
	    if(ii->fileLeft.empty() || ii->fileRight.empty())
	    {
		std::cerr << "ImageViewer: Unable to find files for " << tagBase << std::endl;
		delete ii;
		return;
	    }
	}

	ii->aspectRatio = ConfigManager::getFloat("aspectRatio",tag,-1.0);
	ii->width = ConfigManager::getFloat("width",tag,1000.0);
	ii->scale = ConfigManager::getFloat("scale",tag,1.0);

	float x,y,z;
	x = ConfigManager::getFloat("x",tag,0.0);
	y = ConfigManager::getFloat("y",tag,0.0);
	z = ConfigManager::getFloat("z",tag,0.0);
	ii->position = osg::Vec3(x,y,z);

	float h,p,r;
	h = ConfigManager::getFloat("h",tag,0.0);
	p = ConfigManager::getFloat("p",tag,0.0);
	r = ConfigManager::getFloat("r",tag,0.0);
	ii->rotation = osg::Quat(r,osg::Vec3(0,1.0,0),p,osg::Vec3(1.0,0,0),h,osg::Vec3(0,0,1.0));

	_files.push_back(ii);
	MenuButton * button = new MenuButton(tagBase);
	button->setCallback(this);
	menu->addItem(button);
	_fileButtons.push_back(button);
    }
}
