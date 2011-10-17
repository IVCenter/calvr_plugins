#include "PanoViewLOD.h"

#include <config/ConfigManager.h>
#include <kernel/NodeMask.h>
#include <kernel/PluginHelper.h>

#include <iostream>
#include <cstdlib>

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
    putenv((char*)temp.c_str());

    std::vector<std::string> tagList;
    ConfigManager::getChildren("Plugin.PanoViewLOD.Pans", tagList);

    _panoViewMenu = new SubMenu("PanoView360","PanoView360");
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

    return true;
}

void PanoViewLOD::menuCallback(MenuItem * item)
{
    if(item == _removeButton)
    {
	if(_rightDrawable && _leftDrawable)
	{
	    _leftGeode->removeDrawables(0,_leftGeode->getNumDrawables());
	    _rightGeode->removeDrawables(0,_rightGeode->getNumDrawables());
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

	    _leftDrawable = new PanoDrawableLOD(_pans[i]->leftFiles,_pans[i]->rightFiles,_pans[i]->radius,_pans[i]->mesh,_pans[i]->depth,_pans[i]->size);
	    _rightDrawable = new PanoDrawableLOD(_pans[i]->leftFiles,_pans[i]->rightFiles,_pans[i]->radius,_pans[i]->mesh,_pans[i]->depth,_pans[i]->size);

	    _leftGeode->addDrawable(_leftDrawable);
	    _rightGeode->addDrawable(_rightDrawable);

	    osg::Matrix m,t;
	    m.makeRotate(M_PI/2.0,osg::Vec3(1,0,0));
	    float offset = _pans[i]->height - _floorOffset;
	    t.makeTranslate(osg::Vec3(0,0,offset));
	    _root->setMatrix(m*t);

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
	info->vertFile = vertFile.substr(pos,vertFile.size() - pos);
    }

    std::string fragFile = mxmlElementGetAttr(node, "frag");
    pos = fragFile.find_last_of('/');
    if(pos != std::string::npos)
    {
	info->fragFile = fragFile.substr(pos,fragFile.size() - pos);
    }

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

    return info;
}
