#include "PanoView360.h"

#include <config/ConfigManager.h>
#include <kernel/PluginHelper.h>

#include <mxml.h>

#include <string>

using namespace cvr;
using namespace osg;
using namespace std;

static const string FILES("Plugin.PanoView360.Files");

CVRPLUGIN(PanoView360)

PanoView360* plugin = NULL;

PanoView360::PanoView360()
{

}

bool PanoView360::init()
{
    cerr << "PanoView360 init\n";

    plugin = this;

    _configFile = ConfigManager::getEntry("Plugin.PanoView360.ScreenConfig");

    parseConfig(_configFile);

    _wasinit = 0;
    _deleteWait = false;
    _nextLoad = NULL;

    _cd = NULL;
    _joystickSpin = ConfigManager::getBool("Plugin.PanoView360.JoystickSpin", true);

    _root = new osg::Group();

    _panoViewMenu = new SubMenu("PanoView360","PanoView360");
    _panoViewMenu->setCallback(this);

    _loadMenu = new SubMenu("Load","Load");
    _loadMenu->setCallback(this);

    _panoViewMenu->addItem(_loadMenu);

    _tilesp = new MenuRangeValue("Segments Per Tile", 4, 150, 30);
    _tilesp->setCallback(this);
    _panoViewMenu->addItem(_tilesp);

    _radiusp = new MenuRangeValue("Radius", 10, 100000, 30000);
    _radiusp->setCallback(this);
    _panoViewMenu->addItem(_radiusp);

    _viewanglep = new MenuRangeValue("View Angle: V", 0, 180, 120);
    _viewanglep->setCallback(this);
    _panoViewMenu->addItem(_viewanglep);

    _viewanglepb = new MenuRangeValue("View Angle: H", 0, 360, 120);
    _viewanglepb->setCallback(this);
    _panoViewMenu->addItem(_viewanglepb);

    _camHeightp = new MenuRangeValue("Camera Height", -25000, 25000, 0);
    _camHeightp->setCallback(this);
    _panoViewMenu->addItem(_camHeightp);

    _remove = new MenuButton("Remove");
    _panoViewMenu->addItem(_remove);
    _remove->setCallback(this);


    std::vector<std::string> tagList;
    ConfigManager::getChildren(FILES, tagList);

    for(int i = 0; i < tagList.size(); i++)
    {
	std::string tag = FILES + "." + tagList[i];

	MenuButton* temp = new MenuButton(tagList[i]);
	temp->setCallback(this);
	_menufilelist.push_back(temp);
	struct loadinfo * info = new struct loadinfo;
	info->name = tagList[i];
	info->right_eye_file = ConfigManager::getEntry("reye", tag, "");
	info->left_eye_file = ConfigManager::getEntry("leye", tag, "");
	info->radius = ConfigManager::getFloat("radius", tag, 10000.0);
	info->viewanglev = ConfigManager::getFloat("viewanglev", tag, 120.0);
	info->viewangleh = ConfigManager::getFloat("viewangleh", tag, 360.0);
	info->camHeight = ConfigManager::getFloat("camHeight", tag, 0.0);
	info->segments = ConfigManager::getInt("segments", tag, 25);
	info->texture_size = ConfigManager::getInt("tsize", tag, 1024);
	info->flip = ConfigManager::getInt("flip", tag, 0);
	if(ConfigManager::getInt("sphere", tag, 0))
	{
	    info->shape = SPHERE;
	}
	else
	{
	    info->shape = CYLINDER;
	}

	_pictures.push_back(info);
    }
    
    for(int i = 0; i < _menufilelist.size(); i++)
    {
      _loadMenu->addItem(_menufilelist[i]);
    }

    PluginHelper::addRootMenuItem(_panoViewMenu);
    PluginHelper::getScene()->addChild(_root);    

    cerr << "PanoView360 init done.\n";
    return true;
}


PanoView360::~PanoView360()
{
}

void PanoView360::menuCallback(MenuItem* menuItem)
{
    if(_deleteWait)
    {
	return;
    }
    if(menuItem == _remove)
    {
       // if(root->getNumChildren() != 0)
        //{
        //    root->removeChildren(0, root->getNumChildren());
        //}
	if(_cd != NULL)
	{
	    _cd->deleteTextures();
	    _deleteWait = true;
	    _nextLoad = NULL;
	}
	else
	{
	    if(_root->getNumChildren() != 0)
	    {
	        _root->removeChildren(0, _root->getNumChildren());
	    }
	}
        _wasinit = 0;
        return;
    }

    if(menuItem == _tilesp && _wasinit)
    {
        _cd->setSegmentsPerTexture((int)_tilesp->getValue());
        return;
    }
    if(menuItem == _radiusp && _wasinit)
    {
        _cd->setRadius((int)_radiusp->getValue());
        return;
    }
    if((menuItem == _viewanglep || menuItem == _viewanglepb) && _wasinit)
    {
	_cd->setViewAngle(_viewanglep->getValue(), _viewanglepb->getValue());
	return;
    }

    if(menuItem == _camHeightp && _wasinit)
    {
	_cd->setCamHeight(_camHeightp->getValue());
	return;
    }

    for(int i = 0; i < _menufilelist.size(); i++)
    {
        if(_menufilelist[i] == menuItem)
        {
            //if(root->getNumChildren() != 0)
            //{
                //root->removeChildren(0, root->getNumChildren());
            //}
	    if(_cd != NULL)
	    {
		menuCallback(_remove);
		_nextLoad = menuItem;
		return;
	    }

	    switch(_pictures[i]->shape)
	    {
		case CYLINDER:
		{
		    _cd = new CylinderDrawable(_pictures[i]->radius, _pictures[i]->viewanglev, _pictures[i]->viewangleh, _pictures[i]->camHeight, _pictures[i]->segments, _pictures[i]->texture_size);
		    _cd->setMap(_eyeMap);
		    break;
		}
		case SPHERE:
		{
		    _cd = new SphereDrawable(_pictures[i]->radius, _pictures[i]->viewanglev, _pictures[i]->viewangleh, _pictures[i]->camHeight, _pictures[i]->segments, _pictures[i]->texture_size);
		    _cd->setMap(_eyeMap);
		    break;
		}
		default:
		{
		    cerr << "PanoView360: Unknown shape." << endl;
		    break;
		}
	    }
            _cd->setFlip(_pictures[i]->flip);
            if(_pictures[i]->right_eye_file == "")
            {
                if(_pictures[i]->left_eye_file == "")
                {
                    cerr << "PanoView360: No files listed in config file for " << _pictures[i]->name << endl;
                    _cd->unref();
                    return;
                }
                _cd->setImage(_pictures[i]->left_eye_file);
            }
            else if(_pictures[i]->left_eye_file == "")
            {
                if(_pictures[i]->right_eye_file == "")
                {
                    cerr << "PanoView360: No files listed in config file for " << _pictures[i]->name << endl;
                    _cd->unref();
                    return;
                }
                _cd->setImage(_pictures[i]->right_eye_file);
            }
            else
            {
                _cd->setImage(_pictures[i]->right_eye_file, _pictures[i]->left_eye_file);
            }
            _tilesp->setValue((float)_cd->getSegmentsPerTexture());
            _radiusp->setValue((float)_cd->getRadius());
	    float a, b;
	    _cd->getViewAngle(a, b);
            _viewanglep->setValue(a);
	    _viewanglepb->setValue(b);
            _camHeightp->setValue(_cd->getCamHeight());
            osg::Geode * geo = new osg::Geode();
            geo->addDrawable(_cd);
            _root->addChild(geo);
            _wasinit = 1;
        }
    }
}

/// Called before each frame
void PanoView360::preFrame()
{
    if(_deleteWait)
    {
	if(_cd->deleteDone())
	{
	    if(_root->getNumChildren() != 0)
	    {
	        _root->removeChildren(0, _root->getNumChildren());
	    }
	    _cd = NULL;

	    _deleteWait = false;
	    if(_nextLoad)
	    {
		menuCallback(_nextLoad);
	    }
	}
    }
   if(_cd != NULL && _joystickSpin)
   {
       _cd->updateRotate(PluginHelper::getValuator(0,0));
   } 
}

bool PanoView360::keyEvent(bool, int, int)
{
   if(_cd != NULL)
   {
      _cd->updateRotate(0.6);
   }
   return false;
}

void PanoView360::parseConfig(std::string file)
{
    FILE * fp;
    mxml_node_t * tree;

    //cerr << "Reading file: " << file << endl;

    fp = fopen(file.c_str(), "r");
    if(fp == NULL)
    {
	cerr << "Unable to open file: " << file << std::endl;
	return;
    }

    tree = mxmlLoadFile(NULL, fp,
	    MXML_TEXT_CALLBACK);
    fclose(fp);

    if(tree == NULL)
    {
	cerr << "Unable to parse XML file: " << file << std::endl;
	return;
    }

    mxml_node_t *node;

    for (node = mxmlFindElement(tree, tree, "screen", NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, "screen", NULL, NULL, MXML_DESCEND))
    {
	int vx, vy, context;
	string host, eye;

	vx = atoi(mxmlElementGetAttr(node, "viewportx"));
	vy = atoi(mxmlElementGetAttr(node, "viewporty"));
	context = atoi(mxmlElementGetAttr(node, "context"));
	host = mxmlElementGetAttr(node, "host");
	eye = mxmlElementGetAttr(node, "eye");

	//cerr << "Entry: host: " << host << " vx: " << vx << " vy: " << vy << " context: " << context << " eye: " << eye << endl;

	
	int eyei = 0;
	
	if(eye == "RIGHT")
	{
	    eyei = 1;
	}
	else if(eye == "LEFT")
	{
	    eyei = 2;
	}
	else if(eye == "BOTH")
	{
	    PanoDrawable::firsteye = atoi(mxmlElementGetAttr(node, "firsteye"));
	    eyei = 3;
	}
	else
	{
	    cerr << "Invalid eye entry." << endl;
	}

	_eyeMap[host][context].push_back(pair<pair<int, int>, int >(pair<int, int>(vx, vy), eyei));
    }
}
