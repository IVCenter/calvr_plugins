#include "PanoViewLOD.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/PluginHelper.h>
#include <PluginMessageType.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <mxml.h>
#include <tiffio.h>

#include "sph-cache.hpp"
#include "DiskCache.h"

//#define PRINT_TIMING

using namespace cvr;

CVRPLUGIN(PanoViewLOD)

PanoViewLOD::PanoViewLOD()
{
    _timecount = 0;
    _time = 0;
    _loadRequest = NULL;
}

PanoViewLOD::~PanoViewLOD()
{
    if(sph_cache::_diskCache)
    {
	delete sph_cache::_diskCache;
	sph_cache::_diskCache = NULL;
    }
}

bool PanoViewLOD::init()
{
    _panObject = NULL;

    _defaultConfigDir = ConfigManager::getEntry("value","Plugin.PanoViewLOD.DefaultConfigDir","");

    std::vector<std::string> tagList;
    ConfigManager::getChildren("Plugin.PanoViewLOD.Pans", tagList);

    _panoViewMenu = new SubMenu("PanoViewLOD","PanoViewLOD");
    _panoViewMenu->setCallback(this);

    _loadMenu = new SubMenu("Load","Load");
    _panoViewMenu->addItem(_loadMenu);

    /*_radiusRV = new MenuRangeValue("Radius", 100, 100000, 6000);
    _radiusRV->setCallback(this);
    _panoViewMenu->addItem(_radiusRV);

    _heightRV = new MenuRangeValue("Height", -1000, 10000, 1700);
    _heightRV->setCallback(this);
    _panoViewMenu->addItem(_heightRV);*/

    _removeButton = new MenuButton("Remove");
    _removeButton->setCallback(this);
    _panoViewMenu->addItem(_removeButton);

    _returnButton = new MenuButton("Return");
    _returnButton->setCallback(this);

    for(int i = 0; i < tagList.size(); i++)
    {
	std::string tag = "Plugin.PanoViewLOD.Pans." + tagList[i];
	createLoadMenu(tagList[i], tag, _loadMenu);
    }

    PluginHelper::addRootMenuItem(_panoViewMenu);

    _useDiskCache = ConfigManager::getBool("value","Plugin.PanoViewLOD.DiskCache",true);

    TIFFSetWarningHandler(0);

    return true;
}

void PanoViewLOD::preFrame()
{

#ifdef PRINT_TIMING

    if(_panObject)
    {
	_timecount++;
	_time += PluginHelper::getLastFrameDuration();

	if(_time > 5.0)
	{
	    std::cerr << "FPS: " << _timecount / _time << std::endl;
	    _timecount = 0;
	    _time = 0.0;
	}
    }

#endif
}

bool PanoViewLOD::processEvent(InteractionEvent * event)
{
    return false;
}

void PanoViewLOD::menuCallback(MenuItem * item)
{
    if(item == _removeButton || item == _returnButton)
    {
	/*if(_useDiskCache && sph_cache::_diskCache && sph_cache::_diskCache->isRunning())
	{
	    sph_cache::_diskCache->stop();
	}*/

	removePan();
    }
    for(int i = 0; i < _panButtonList.size(); i++)
    {
	if(item == _panButtonList[i])
	{
	    if(_useDiskCache && !sph_cache::_diskCache)
	    {
		//std::cerr << "Creating cache in plugin." << std::endl;
		sph_cache::_diskCache = new DiskCache(cvr::ConfigManager::getInt("value","Plugin.PanoViewLOD.DiskCacheSize",256));
		sph_cache::_diskCache->start();
	    }

	    /*if(_useDiskCache && sph_cache::_diskCache && !sph_cache::_diskCache->isRunning())
	    {
		sph_cache::_diskCache->start();
	    }*/

	    removePan();

	    _panObject = new PanoViewObject(_panButtonList[i]->getText(),_pans[i]->leftFiles,_pans[i]->rightFiles,_pans[i]->radius,_pans[i]->mesh,_pans[i]->depth,_pans[i]->size,_pans[i]->height,_pans[i]->vertFile,_pans[i]->fragFile);

	    PluginHelper::registerSceneObject(_panObject,"PanoViewLOD");
	    _panObject->attachToScene();

	    if(_loadRequest)
	    {
		_panObject->addMenuItem(_returnButton);
	    }

	    break;
	}
    }
}

void PanoViewLOD::message(int type, char *&data, bool collaborative)
{
    if(type == PAN_LOAD_REQUEST)
    {
	if(collaborative)
	{
	    return;
	}

	PanLoadRequest * plr = (PanLoadRequest*)data;

	int i;
	for(i = 0; i < _panButtonList.size(); i++)
	{
	    if(plr->name == _panButtonList[i]->getText())
	    {
		break;
	    }
	}

	if(i == _panButtonList.size())
	{
	    std::cerr << "PanoViewLOD message load: no pan named " << plr->name << std::endl;
	    plr->loaded = false;
	    return;
	}

	removePan();

	_loadRequest = new PanLoadRequest;
	*_loadRequest = *plr;

	menuCallback(_panButtonList[i]);
	plr->loaded = true;
	return;
    }

    if(type == PAN_HEIGHT_REQUEST)
    {
	if(collaborative)
	{
	    return;
	}

	PanHeightRequest * phr = (PanHeightRequest*)data;

	int i;
	for(i = 0; i < _panButtonList.size(); i++)
	{
	    if(phr->name == _panButtonList[i]->getText())
	    {
		break;
	    }
	}

	if(i == _panButtonList.size())
	{
	    return;
	}

	phr->height = _pans[i]->height - ConfigManager::getFloat("value","Plugin.PanoViewLOD.FloorOffset",1500.0) + DEFAULT_PAN_HEIGHT;
    }

    if(type == PAN_SET_ALPHA)
    {
	float alpha = *((float*)data);

	if(_panObject)
	{
	    _panObject->setAlpha(alpha);
	}
    }

    if(type == PAN_UNLOAD)
    {
	removePan();
    }
}

void PanoViewLOD::createLoadMenu(std::string tagBase, std::string tag, SubMenu * menu)
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
	MenuButton* temp = new MenuButton(tagBase);
	temp->setCallback(this);

	bool fl,fr,fxml;
	ConfigManager::getEntry("leftImage",tag,"",&fl);
	ConfigManager::getEntry("rightImage",tag,"",&fr);
	ConfigManager::getEntry("xmlFile",tag,"",&fxml);

	PanInfo * info = NULL;

	if(fl && fr)
	{
	    info = new PanInfo;
	    info->leftFiles.push_back(ConfigManager::getEntry("leftImage",tag,""));
	    info->rightFiles.push_back(ConfigManager::getEntry("rightImage",tag,""));
	    info->depth = ConfigManager::getInt("depth",tag,3);
	    info->mesh = ConfigManager::getInt("mesh",tag,16);
	    info->size = ConfigManager::getInt("size",tag,512);
	    info->vertFile = ConfigManager::getEntry("vertFile",tag,"sph-zoomer.vert");
	    info->fragFile = ConfigManager::getEntry("fragFile",tag,"sph-render.frag");
	    info->height = ConfigManager::getFloat("height",tag,0.0);
	}
	else if(fxml)
	{
	    info = loadInfoFromXML(ConfigManager::getEntry("xmlFile",tag,""));
	}
	else
	{
	    info = loadInfoFromXML(tagBase + ".xml");
	}

	if(info)
	{
	    //put here for now since the values are not in the xml files
	    info->radius = ConfigManager::getFloat("radius",tag,6000);
	    _panButtonList.push_back(temp);
	    _pans.push_back(info);
	    menu->addItem(temp);
	}
	else
	{
	    std::cerr << "Unable to find pan info for tag: " << tag << std::endl;
	    delete temp;
	}
    }
}

PanoViewLOD::PanInfo * PanoViewLOD::loadInfoFromXML(std::string file)
{
    FILE * fp;
    mxml_node_t * tree;

    fp = fopen(file.c_str(), "r");
    if(!fp)
    {
	fp = fopen((_defaultConfigDir + "/" + file).c_str(),"r");
	if(!fp)
	{
	    return NULL;
	}
    }

    tree = mxmlLoadFile(NULL, fp,
	    MXML_TEXT_CALLBACK);
    fclose(fp);

    if(tree == NULL)
    {
	std::cerr << "Unable to parse XML file: " << file << std::endl;
	return NULL;
    }

    bool mono = false;

    mxml_node_t * node = mxmlFindElement(tree, tree, "panorama", NULL, NULL, MXML_DESCEND);

    if(!node)
    {
	std::cerr << "No panorama tag in file: " << file << std::endl;
	return NULL;
    }

    PanInfo * info = new PanInfo;

    if(atoi(mxmlElementGetAttr(node, "channels")) != 2)
    {
	mono = true;
    }

    info->depth = atoi(mxmlElementGetAttr(node, "depth"));
    info->mesh = atoi(mxmlElementGetAttr(node, "mesh"));
    info->size = atoi(mxmlElementGetAttr(node, "size"));
    std::string vertFile = mxmlElementGetAttr(node, "vert");

    const char * height = mxmlElementGetAttr(node, "height");
    if(height)
    {
	info->height = atof(height);
    }

    size_t pos = vertFile.find_last_of('/');
    if(pos != std::string::npos)
    {
	vertFile = vertFile.substr(pos,vertFile.size() - pos);
    }
    info->vertFile = vertFile;

    //std::cerr << "VertFile: " << info->vertFile << std::endl;

    std::string fragFile = mxmlElementGetAttr(node, "frag");
    pos = fragFile.find_last_of('/');
    if(pos != std::string::npos)
    {
	fragFile = fragFile.substr(pos,fragFile.size() - pos);
    }
    info->fragFile = fragFile;

    //std::cerr << "FragFile: " << info->fragFile << std::endl;

    mxml_node_t * node2;

    for(node2 = mxmlFindElement(node, node, "image", NULL, NULL, MXML_DESCEND); node2 != NULL; node2 = mxmlFindElement(node2, node, "image", NULL, NULL, MXML_DESCEND))
    {
	int channel = atoi(mxmlElementGetAttr(node2, "channel"));
	std::string pfile = mxmlElementGetAttr(node2, "file");
	if(channel == 0)
	{
	    info->leftFiles.push_back(pfile);
	    if(mono)
	    {
		info->rightFiles.push_back(pfile);
	    }
	}
	else if(channel == 1 && !mono)
	{
	    info->rightFiles.push_back(pfile);
	}
    }

    if(info->leftFiles.size() != info->rightFiles.size())
    {
	std::cerr << "Unmatched left/right files in: " << file << std::endl;
	delete info;
	return NULL;
    }

    if(0)
    {
        std::cerr << "File: " << file << std::endl;
        std::cerr << "LeftFiles:" << std::endl;
        for(int i = 0; i < info->leftFiles.size(); i++)
        {
            std::cerr << info->leftFiles[i] << std::endl;
        }
        std::cerr << "RightFiles:" << std::endl;
        for(int i = 0; i < info->rightFiles.size(); i++)
        {
            std::cerr << info->rightFiles[i] << std::endl;
        }
        std::cerr << "Depth: " << info->depth << std::endl;
        std::cerr << "Mesh: " << info->mesh << std::endl;
        std::cerr << "Size: " << info->size << std::endl;
        std::cerr << "VertFile: " << info->vertFile << std::endl;
        std::cerr << "FragFile: " << info->fragFile << std::endl;
    }

    return info;
}

void PanoViewLOD::updateZoom(osg::Matrix & mat)
{
    /*osg::Matrix m = osg::Matrix::inverse(_root->getMatrix());
    osg::Vec3 dir(0,1,0);
    osg::Vec3 point(0,0,0);
    dir = dir * mat * m;
    point = point * mat * m;
    dir = dir - point;
    dir.normalize();

    if(_leftDrawable)
    {
	_leftDrawable->setZoom(dir,pow(10.0, _currentZoom));
    }
    else
    {
	_rightDrawable->setZoom(dir,pow(10.0, _currentZoom));
    }

    _lastZoomMat = mat;*/
}

void PanoViewLOD::removePan()
{
    if(_panObject)
    {
	if(_loadRequest)
	{
	    PluginHelper::sendMessageByName(_loadRequest->plugin,_loadRequest->pluginMessageType,(char*)NULL);
	    delete _loadRequest;
	    _loadRequest = NULL;
	}
	else
	{
	    _panObject->detachFromScene();
	    PluginHelper::unregisterSceneObject(_panObject);
	    delete _panObject;
	    _panObject = NULL;
	}
    }
}
