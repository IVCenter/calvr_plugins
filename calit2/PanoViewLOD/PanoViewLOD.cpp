#include "PanoViewLOD.h"

#include <config/ConfigManager.h>
#include <kernel/NodeMask.h>
#include <kernel/PluginHelper.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <mxml.h>

using namespace cvr;

CVRPLUGIN(PanoViewLOD)

PanoViewLOD::PanoViewLOD()
{
}

PanoViewLOD::~PanoViewLOD()
{
}

bool PanoViewLOD::init()
{
    _root = new osg::MatrixTransform();
    _leftGeode = new osg::Geode();
    _rightGeode = new osg::Geode();

    _root->addChild(_leftGeode);
    _root->addChild(_rightGeode);

    _leftGeode->setNodeMask(_leftGeode->getNodeMask() & (~CULL_MASK_RIGHT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK_LEFT));
    _rightGeode->setNodeMask(_rightGeode->getNodeMask() & (~CULL_MASK));

    _rightDrawable = NULL;
    _leftDrawable = NULL;

    _defaultConfigDir = ConfigManager::getEntry("value","Plugin.PanoViewLOD.DefaultConfigDir","");
    _imageSearchPath = ConfigManager::getEntry("value","Plugin.PanoViewLOD.ImageSearchPath","");
    _floorOffset = ConfigManager::getFloat("value","Plugin.PanoViewLOD.FloorOffset",0);

    std::string temp("PANOPATH=");
    temp = temp + _imageSearchPath;

    char * carray = new char[temp.size()+1];
    
    strcpy(carray,temp.c_str());

    putenv(carray);

    std::vector<std::string> tagList;
    ConfigManager::getChildren("Plugin.PanoViewLOD.Pans", tagList);

    _panoViewMenu = new SubMenu("PanoViewLOD","PanoViewLOD");
    _panoViewMenu->setCallback(this);

    _loadMenu = new SubMenu("Load","Load");
    _panoViewMenu->addItem(_loadMenu);

    _removeButton = new MenuButton("Remove");
    _removeButton->setCallback(this);
    _panoViewMenu->addItem(_removeButton);

    for(int i = 0; i < tagList.size(); i++)
    {
	std::string tag = "Plugin.PanoViewLOD.Pans." + tagList[i];
	createLoadMenu(tagList[i], tag, _loadMenu);
    }

    PluginHelper::addRootMenuItem(_panoViewMenu);

    return true;
}

bool PanoViewLOD::processEvent(InteractionEvent * event)
{
    if(event->asTrackedButtonEvent())
    {
	TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
	if(_rightDrawable || _leftDrawable)
	{
	    if(tie->getButton() == 2 && tie->getInteraction() == BUTTON_DOWN)
	    {
		if(_rightDrawable)
		{
		    _rightDrawable->next();
		}
		if(_leftDrawable)
		{
		    _leftDrawable->next();
		}
		return true;
	    }
	    if(tie->getButton() == 3 && tie->getInteraction() == BUTTON_DOWN)
	    {
		if(_rightDrawable)
		{
		    _rightDrawable->previous();
		}
		if(_leftDrawable)
		{
		    _leftDrawable->previous();
		}
		return true;
	    }
	    if(tie->getButton() == 0 && tie->getInteraction() == BUTTON_DOWN)
	    {
		//_zoomActive = true;
		//_currentZoom = 0.0;
                osg::Matrix m;
                m.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
		osg::Vec3 dir(0,1,0);
		dir = dir * tie->getTransform();
		dir = dir - tie->getTransform().getTrans();
                dir = dir * m;
		dir.normalize();

		if(_leftDrawable)
		{
		    _leftDrawable->setZoom(dir,pow(10.0, _currentZoom));
		}
		else
		{
		    _rightDrawable->setZoom(dir,pow(10.0, _currentZoom));
		}

		return true;
	    }
	    if(tie->getButton() == 0 && (tie->getInteraction() == BUTTON_DRAG || tie->getInteraction() == BUTTON_UP))
	    {
		float val = PluginHelper::getValuator(0,0);
		if(fabs(val) > 0.25)
		{
		    _currentZoom += val * PluginHelper::getLastFrameDuration() * 0.25;
		    if(_currentZoom < -2.0) _currentZoom = -2.0;
		    if(_currentZoom > 0.5) _currentZoom = 0.5;
		}
                
                osg::Matrix m;
                m.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
		osg::Vec3 dir(0,1,0);
		dir = dir * tie->getTransform();
		dir = dir - tie->getTransform().getTrans();
                dir = dir * m;
		dir.normalize();

		if(_leftDrawable)
		{
		    _leftDrawable->setZoom(dir,pow(10.0, _currentZoom));
		}
		else
		{
		    _rightDrawable->setZoom(dir,pow(10.0, _currentZoom));
		}
		return true;
	    }
	    if(tie->getButton() == 4 && tie->getInteraction() == BUTTON_DOWN)
	    {
		_currentZoom = 0.0;

		if(_leftDrawable)
		{
		    _leftDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
		}
		else
		{
		    _rightDrawable->setZoom(osg::Vec3(0,1,0),pow(10.0, _currentZoom));
		}

		return true;
	    }
	}
    }
    return false;
}

void PanoViewLOD::menuCallback(MenuItem * item)
{
    if(item == _removeButton)
    {
	if(_rightDrawable && _leftDrawable)
	{
            _leftDrawable->cleanup();
	    _leftGeode->removeDrawables(0,_leftGeode->getNumDrawables());
	    _rightGeode->removeDrawables(0,_rightGeode->getNumDrawables());
            _leftDrawable = _rightDrawable = NULL;
	}
	PluginHelper::getScene()->removeChild(_root);
    }
    for(int i = 0; i < _panButtonList.size(); i++)
    {
	if(item == _panButtonList[i])
	{
	    if(_rightDrawable && _leftDrawable)
	    {
		menuCallback(_removeButton);
	    }

	    _leftDrawable = new PanoDrawableLOD(_pans[i]->leftFiles,_pans[i]->rightFiles,_pans[i]->radius,_pans[i]->mesh,_pans[i]->depth,_pans[i]->size,_pans[i]->vertFile,_pans[i]->fragFile);
	    _rightDrawable = new PanoDrawableLOD(_pans[i]->leftFiles,_pans[i]->rightFiles,_pans[i]->radius,_pans[i]->mesh,_pans[i]->depth,_pans[i]->size,_pans[i]->vertFile,_pans[i]->fragFile);

	    _leftGeode->addDrawable(_leftDrawable);
	    _rightGeode->addDrawable(_rightDrawable);

	    osg::Matrix m,t;
	    m.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
	    float offset = _pans[i]->height - _floorOffset;
	    t.makeTranslate(osg::Vec3(0,0,offset));
	    _root->setMatrix(m);

	    PluginHelper::getScene()->addChild(_root);

	    break;
	}
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
	    //info->height = ConfigManager::getFloat("height",tag,1500);
	    //info->radius = ConfigManager::getFloat("radius",tag,6000);
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
	    info->height = ConfigManager::getFloat("height",tag,1500);
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
