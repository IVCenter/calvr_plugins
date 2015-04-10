#include "PointsOOC.h"

#include <PluginMessageType.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iterator>
#include <math.h>

// OSG:
#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/Shader>
#include <osg/AlphaFunc>
#include <osg/StateAttribute>
#include <osg/PolygonMode>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/Texture2D>

#include <osg/Vec3d>
#include <osg/MatrixTransform>
#include <map>
#include <limits>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

using namespace std;
using namespace cvr;
using namespace osg;

CVRPLUGIN(PointsOOC)

//constructor
PointsOOC::PointsOOC() : FileLoadCallback("osga")
{

}

void PointsOOC::adjustLODScale(float lower, float upper, float interval)
{
	if( _loadedPoints.size() != 0 )
	{
    	osg::FrameStamp* fs = CVRViewer::instance()->getFrameStamp();

    	if( fs )
    	{
        	// update frame rate every 0.025 seconds (default setting)
        	if( (fs->getReferenceTime() - _startTime) > interval )
        	{
        	    //frameRate =  (fs->getReferenceTime() - _startTime ) / (fs->getFrameNumber() - _currentFrame); 
        	    float frameRate =  (float)(fs->getFrameNumber() - _currentFrame) / (fs->getReferenceTime() - _startTime);

				// check if frame rate is outside bounds
            	if( frameRate < lower )
            	{	
                	// increae lodScale
                	CVRViewer::instance()->getCamera()->setLODScale(CVRViewer::instance()->getCamera()->getLODScale() * 1.1);
                	//std::cerr << "Scale increased to: " << CVRViewer::instance()->getCamera()->getLODScale() << std::endl;
            	}
            	else if( frameRate > upper && CVRViewer::instance()->getCamera()->getLODScale() > 1.0 )
            	{
                	//decrease lodScale
                	CVRViewer::instance()->getCamera()->setLODScale(CVRViewer::instance()->getCamera()->getLODScale() / 1.1);
                	//std::cerr << "Scale decreased to: " << CVRViewer::instance()->getCamera()->getLODScale() << std::endl;
            	}

        	    // update new values 
        	    _startTime = fs->getReferenceTime();
        	    _currentFrame = fs->getFrameNumber();
        	}
    	}
	}
}


bool PointsOOC::loadFile(std::string filename)
{
    osg::ref_ptr<Node> points = osgDB::readNodeFile(filename);

	if( points.valid() )
	{

        osg::StateSet* state = points->getOrCreateStateSet();

    	// check to make sure that the osga is a point cloud (check for uniform)
		osg::Uniform* pointUniform = state->getUniform("point_size");
    	if( pointUniform != NULL )
    	{

			// check if see if this is the first set to be loaded
			if( _loadedPoints.size() == 0 )
			{
				_startTime = CVRViewer::instance()->getFrameStamp()->getReferenceTime();
				_currentFrame = CVRViewer::instance()->getFrameStamp()->getFrameNumber();
			} 
		
			// get fileName
			std::string name = osgDB::getStrippedName(filename); 

			// add stream to the scene
			SceneObject * so = new SceneObject(name,false,false,false,true,true);
			PluginHelper::registerSceneObject(so,"PointsOOC");
			so->addChild(points.get());
			so->attachToScene();
			so->setNavigationOn(true);
			so->addMoveMenuItem();
			so->addNavigationMenuItem();
		    so->setShowBounds(false);

            // add submenu to control point size options
            SubMenu * sm = new SubMenu("Point Size Options");
            so->addMenuItem(sm);
		    _subMenuMap[so] = sm;

			// add menu items for controlling the points
		    MenuCheckbox * cb = new MenuCheckbox("Show Bounds", false);
		    cb->setCallback(this);
		    so->addMenuItem(cb);
		    _boundsMap[so] = cb;
		    
            cb = new MenuCheckbox("Enable Shader", true);
		    cb->setCallback(this);
		    so->addMenuItem(cb);
		    _shaderMap[so] = cb;
		    
			// get point size
			float shaderSize;
			pointUniform->get(shaderSize);

			// access the alpha
	    	osg::Uniform* alphaUniform = state->getUniform("global_alpha");

            // doesnt exist in table create a new entry
            if( _locInit.find(name) == _locInit.end() )
            {
                Loc l;
                l.pos = osg::Matrix::identity();
                l.pointSize = 1.0;
                l.shaderSize = shaderSize;
                l.shaderEnabled = true;
                l.pointAlpha = 1.0;
                l.pointFunc[0] = 1.0;
                l.pointFunc[1] = 0.05;
                l.pointFunc[2] = 0.0;
                _locInit[name] = l;
            }

            _locInit[name].shaderStateset = state;
            
            // create point stateset
            _locInit[name].pointStateset = new osg::StateSet();

            _locInit[name].pointStateset->setMode(GL_POINT_SMOOTH, osg::StateAttribute::ON);

	    _locInit[name].pointStateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
	    _locInit[name].pointStateset->setMode(GL_BLEND, osg::StateAttribute::ON);
	    _locInit[name].pointStateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

            osg::Point* point = new osg::Point;
            point->setDistanceAttenuation(osg::Vec3(_locInit[name].pointFunc[0], _locInit[name].pointFunc[1], _locInit[name].pointFunc[2]));
            point->setSize(_locInit[name].pointSize);
            point->setMinSize(1.0);
            point->setMaxSize(100.0);
            _locInit[name].pointStateset->setAttribute(point);

            osg::PolygonMode *pm = new osg::PolygonMode( osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::POINT );
            _locInit[name].pointStateset->setAttribute(pm);

            // read it from the saved file
	    pointUniform->set(_locInit[name].shaderSize);
	    alphaUniform->set(_locInit[name].pointAlpha);

            // point variation
            float variation = _locInit[name].shaderSize * 0.75;

        	MenuRangeValue * mrv = new MenuRangeValue("Point Size Shader", _locInit[name].shaderSize - variation, 
                                                _locInit[name].shaderSize + variation, _locInit[name].shaderSize);
	    	mrv->setCallback(this);
        	_sliderShaderSizeMap[so] = mrv;
	    	
        	mrv = new MenuRangeValue("Point Size", 0.01, 50.0, _locInit[name].pointSize);
            mrv->setCallback(this);
        	_sliderPointSizeMap[so] = mrv;
        	
            mrv = new MenuRangeValue("Point Func", 0.0, 0.9, _locInit[name].pointFunc[1]);
            mrv->setCallback(this);
        	_sliderPointFuncMap[so] = mrv;

			mrv = new MenuRangeValue("Alpha", 0.0, 1.0, _locInit[name].pointAlpha);
	    	mrv->setCallback(this);
        	sm->addItem(mrv);
        	_sliderAlphaMap[so] = mrv;

            // check if shader enabled
            if( !_locInit[name].shaderEnabled )
            {
                _shaderMap[so]->setValue(false);
                points->setStateSet(_locInit[name].pointStateset);

                sm->addItem(_sliderPointSizeMap[so]);
                sm->addItem(_sliderPointFuncMap[so]);
            }
            else
            {
                sm->addItem(_sliderShaderSizeMap[so]);
            }

	    // set default lighting and blend modes
            points->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
            points->getStateSet()->setMode(GL_BLEND,osg::StateAttribute::OFF);

            //check pointAlpha
            if( _locInit[name].pointAlpha != 1.0 )
            {
                points->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                points->getStateSet()->setMode(GL_BLEND,osg::StateAttribute::ON);
                points->getStateSet()->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );
            }

        	bool nav;
        	nav = so->getNavigationOn();
        	so->setNavigationOn(false);

        	so->setTransform(_locInit[name].pos);
        	so->setNavigationOn(nav);

            // add default buttons
			MenuButton* mb = new MenuButton("Save");
	    	mb->setCallback(this);
	    	so->addMenuItem(mb);
	    	_saveMap[so] = mb;			

			mb = new MenuButton("Delete");
	    	mb->setCallback(this);
	    	so->addMenuItem(mb);
	    	_deleteMap[so] = mb;
			
			// to list of objects
			_loadedPoints.push_back(so);

			return true;
		}
	}

    return false;
}


void PointsOOC::removeAll()
{
    std::cerr << "Remove all called\n";

    std::map<SceneObject*,cvr::MenuRangeValue*>::iterator it;
    for(it = _sliderPointSizeMap.begin(); it != _sliderPointSizeMap.end(); it++)
    {
        delete it->second;
    }
    
    for(it = _sliderShaderSizeMap.begin(); it != _sliderShaderSizeMap.end(); it++)
    {
        delete it->second;
    }
    
    for(it = _sliderPointFuncMap.begin(); it != _sliderPointFuncMap.end(); it++)
    {
        delete it->second;
    }
  
    for(it = _sliderAlphaMap.begin(); it != _sliderAlphaMap.end(); it++)
    {
        delete it->second;
    }
  
    std::map<SceneObject*,cvr::MenuButton*>::iterator itt;
    for(itt = _deleteMap.begin(); itt != _deleteMap.end(); itt++)
    {
        delete itt->second;
    }
    
    for(itt = _saveMap.begin(); itt != _saveMap.end(); itt++)
    {
        delete itt->second;
    }
    
    std::map<SceneObject*,cvr::MenuCheckbox*>::iterator ittt;
    for(ittt = _boundsMap.begin(); ittt != _boundsMap.end(); ittt++)
    {
        delete ittt->second;
    }
    
    for(ittt = _shaderMap.begin(); ittt != _shaderMap.end(); ittt++)
    {
        delete ittt->second;
    }
    
    std::map<SceneObject*,cvr::SubMenu*>::iterator itttt;
    for(itttt = _subMenuMap.begin(); itttt != _subMenuMap.end(); itttt++)
    {
        delete itttt->second;
    }
    
    _sliderPointSizeMap.clear();
    _sliderShaderSizeMap.clear();
    _sliderPointFuncMap.clear();
    _sliderAlphaMap.clear();
    _boundsMap.clear();
    _shaderMap.clear();
    _subMenuMap.clear();
    _saveMap.clear();
    _deleteMap.clear();

    for(int i = 0; i < _loadedPoints.size(); i++)
    {
        delete _loadedPoints[i];
    }

    _loadedPoints.clear();

    return;
}


void PointsOOC::menuCallback(MenuItem* menuItem)
{
    // load file
    for(int i = 0; i < _menuFileList.size(); i++)
    {
	    if(_menuFileList[i] == menuItem)
        {
            //removeAll();

			if( !osgDB::fileExists(_filePaths[i]) )
            {
                std::cerr << "PointsOOC: file not found: " << 
                    _filePaths[i] << endl;
                return;
            }

            loadFile(_filePaths[i]);
        }
    }


    // shader slider
    for(std::map<cvr::SceneObject*,MenuRangeValue*>::iterator it = _sliderShaderSizeMap.begin(); it != _sliderShaderSizeMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            float pointSize = it->second->getValue();
			
            // get access to uniform and adjust the size
			osg::Uniform* sizeUniform = _locInit[it->first->getName()].shaderStateset->getUniform("point_size");
			if( sizeUniform )
			{
				sizeUniform->set(pointSize);
            }
            break;
        }
    }

    // point size
    for(std::map<cvr::SceneObject*,MenuRangeValue*>::iterator it = _sliderPointSizeMap.begin(); it != _sliderPointSizeMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            osg::Point* point = dynamic_cast<osg::Point* >(_locInit[it->first->getName()].pointStateset->getAttribute(osg::StateAttribute::POINT));
            if( point )
            {
                point->setSize(it->second->getValue());
			}
			break;
        }
    }
    
    // point func
    for(std::map<cvr::SceneObject*,MenuRangeValue*>::iterator it = _sliderPointFuncMap.begin(); it != _sliderPointFuncMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            _locInit[it->first->getName()].pointFunc[1] =  it->second->getValue();

            osg::Point* point = dynamic_cast<osg::Point* >(_locInit[it->first->getName()].pointStateset->getAttribute(osg::StateAttribute::POINT));
            if( point )
            {
                point->setDistanceAttenuation(osg::Vec3(_locInit[it->first->getName()].pointFunc[0], _locInit[it->first->getName()].pointFunc[1], _locInit[it->first->getName()].pointFunc[2]));
			}
			break;
        }
    }

	// alpha slider
    for(std::map<cvr::SceneObject*,MenuRangeValue*>::iterator it = _sliderAlphaMap.begin(); it != _sliderAlphaMap.end(); it++)
    {
        if(menuItem == it->second)
        {
			float pointAlpha = it->second->getValue();
            		
			// get access to uniform and adjust the size
			osg::Uniform* alphaUniform = _locInit[it->first->getName()].shaderStateset->getUniform("global_alpha");
			if( alphaUniform )
			{
				alphaUniform->set(pointAlpha);
            }

            // set point alpha
            osg::StateSet* sstate = _locInit[it->first->getName()].shaderStateset;
            osg::StateSet* pstate = _locInit[it->first->getName()].pointStateset;
            if( pointAlpha == 1.0 )
            {
                    sstate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                    sstate->setMode(GL_BLEND,osg::StateAttribute::OFF);
                    sstate->setRenderingHint( osg::StateSet::OPAQUE_BIN );
                    
                    pstate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
                    //pstate->setMode(GL_BLEND,osg::StateAttribute::OFF);
                    pstate->setRenderingHint( osg::StateSet::OPAQUE_BIN );
                    pstate->removeAttribute(osg::StateAttribute::ALPHAFUNC);
            }
            else
            {
                    sstate->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                    sstate->setMode(GL_BLEND,osg::StateAttribute::ON);
                    sstate->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );
                    
                    pstate->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                    //pstate->setMode(GL_BLEND,osg::StateAttribute::ON);
                    pstate->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );
                    pstate->setAttribute(new osg::AlphaFunc(osg::AlphaFunc::GREATER, 1.0 - pointAlpha));
            }

			break;
        }
    }


    // show bounding box
    for(std::map<cvr::SceneObject*,MenuCheckbox*>::iterator it = _boundsMap.begin(); 
        it != _boundsMap.end(); it++)
    {
        if(menuItem == it->second)
        {
        	it->first->setShowBounds(it->second->getValue());
			break;
        }
    }
    
    for(std::map<cvr::SceneObject*,MenuCheckbox*>::iterator it = _shaderMap.begin(); 
        it != _shaderMap.end(); it++)
    {
        if(menuItem == it->second)
        {

            SubMenu* sm = _subMenuMap[it->first];

            if( it->second->getValue() )
            {
                it->first->getChildNode(0)->setStateSet(_locInit[it->first->getName()].shaderStateset);

                // remove items
                sm->removeItem(_sliderPointSizeMap[it->first]);
                sm->removeItem(_sliderPointFuncMap[it->first]);

                // add applicable item
                sm->addItem(_sliderShaderSizeMap[it->first]);

            }
            else
            {
                it->first->getChildNode(0)->setStateSet(_locInit[it->first->getName()].pointStateset);
                
                // remove items
                sm->removeItem(_sliderShaderSizeMap[it->first]);
               
                // ad applicable item 
                sm->addItem(_sliderPointSizeMap[it->first]);
                sm->addItem(_sliderPointFuncMap[it->first]);
            }
			break;
        }
    }
    
    // check map for a delete
    for(std::map<cvr::SceneObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(_sliderPointSizeMap.find(it->first) != _sliderPointSizeMap.end())
            {
                delete _sliderPointSizeMap[it->first];
                _sliderPointSizeMap.erase(it->first);
            }
            
            if(_sliderShaderSizeMap.find(it->first) != _sliderShaderSizeMap.end())
            {
                delete _sliderShaderSizeMap[it->first];
                _sliderShaderSizeMap.erase(it->first);
            }
            
            if(_sliderPointFuncMap.find(it->first) != _sliderPointFuncMap.end())
            {
                delete _sliderPointFuncMap[it->first];
                _sliderPointFuncMap.erase(it->first);
            }
			
			if(_sliderAlphaMap.find(it->first) != _sliderAlphaMap.end())
            {
                delete _sliderAlphaMap[it->first];
                _sliderAlphaMap.erase(it->first);
            }

			if(_saveMap.find(it->first) != _saveMap.end())
	    	{
				delete _saveMap[it->first];
				_saveMap.erase(it->first);
	    	}
			
            if(_subMenuMap.find(it->first) != _subMenuMap.end())
	    	{
				delete _subMenuMap[it->first];
				_subMenuMap.erase(it->first);
	    	}


            for(std::vector<cvr::SceneObject*>::iterator delit = _loadedPoints.begin(); delit != _loadedPoints.end(); delit++)
            {
                if((*delit) == it->first)
                {
                    _loadedPoints.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            break;
        }
    }

    // save transform, pointScale, shader and pointAlpha
    for(std::map<cvr::SceneObject*, MenuButton*>::iterator it = _saveMap.begin(); it != _saveMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            for (int i = 0; i < _loadedPoints.size(); ++i)
            {
                bool nav;
                nav = _loadedPoints[i]->getNavigationOn();
                _loadedPoints[i]->setNavigationOn(false);

				_locInit[_loadedPoints[i]->getName()].pos = _loadedPoints[i]->getTransform();
				_locInit[_loadedPoints[i]->getName()].shaderSize = _sliderShaderSizeMap[_loadedPoints[i]]->getValue();
				_locInit[_loadedPoints[i]->getName()].pointSize = _sliderPointSizeMap[_loadedPoints[i]]->getValue();
				_locInit[_loadedPoints[i]->getName()].pointFunc[1] = _sliderPointFuncMap[_loadedPoints[i]]->getValue();
				_locInit[_loadedPoints[i]->getName()].pointAlpha = _sliderAlphaMap[_loadedPoints[i]]->getValue();				
				_locInit[_loadedPoints[i]->getName()].shaderEnabled = _shaderMap[_loadedPoints[i]]->getValue();				
			 
                _loadedPoints[i]->setNavigationOn(nav);
            }
            writeConfigFile();
        }
    }

    // check for removeAll
    if( menuItem == _removeButton )
    {
        removeAll();
    }

}


// intialize
bool PointsOOC::init()
{
  cerr << "PointsOOC::PointsOOC" << endl;

  // enable osg debugging
  //osg::setNotifyLevel( osg::INFO );
  
  
  _mainMenu = new SubMenu("PointsOOC", "PointsOOC");
  _mainMenu->setCallback(this);

  _loadMenu = new SubMenu("Load","Load");
  _loadMenu->setCallback(this);
  _mainMenu->addItem(_loadMenu);

  _removeButton = new MenuButton("Remove All");
  _removeButton->setCallback(this);
  _mainMenu->addItem(_removeButton);

  MenuSystem::instance()->addMenuItem(_mainMenu);


  vector<string> list;

  string configBase = "Plugin.PointsOOC.Files";

  ConfigManager::getChildren(configBase, list);

  for(int i = 0; i < list.size(); i++)
  {
	MenuButton * button = new MenuButton(list[i]);
	button->setCallback(this);
	_loadMenu->addItem(button);
    _menuFileList.push_back(button);

	std::string path = ConfigManager::getEntry("value", 
        configBase + "." + list[i], "");
    _filePaths.push_back(path);
    //std::cout << path << std::endl;
  }


  // load saved initial scales and locations
  _configPath = ConfigManager::getEntry("Plugin.PointsOOC.ConfigDir");

  ifstream cfile;
  cfile.open((_configPath + "/PointsOOCInit.cfg").c_str(), ios::in);

  if(!cfile.fail())
  {
    string line;
    while(!cfile.eof())
    {

	   Loc l;
       l.pointFunc[0] = 1.0;
       l.pointFunc[1] = 0.0;

       char name[150];
       cfile >> name;
       if(cfile.eof())
       {
         break;
       }
	   cfile >> l.pointSize;
	   cfile >> l.shaderSize;
	   cfile >> l.pointFunc[2];
	   cfile >> l.pointAlpha;
       cfile >> l.shaderEnabled;
       for(int i = 0; i < 4; i++)
       {
         for(int j = 0; j < 4; j++)
         {
           cfile >> l.pos(i, j);
         }
       }

       _locInit[string(name)] = l;
    }
  }
  cfile.close();


  return true;
}

// this is called if the plugin is removed at runtime
PointsOOC::~PointsOOC()
{
   fprintf(stderr,"PointsOOC::~PointsOOC\n");
}

void PointsOOC::preFrame()
{

	// call the framerate adjuster
    adjustLODScale();
	
}

void PointsOOC::message(int type, char *&data, bool collaborative)
{
    if(type == POINTS_LOAD_REQUEST)
    {
        if(collaborative)
        {
            return;
        }

        PointsLoadInfo * pli = (PointsLoadInfo*) data;
        if(!pli->group)
        {
            return;
        }

        loadFile(pli->file);
    }
}

void PointsOOC::writeConfigFile()
{
    if (!cvr::ComController::instance()->isMaster())
    {
        return;
    }

    ofstream cfile;
    cfile.open((_configPath + "/PointsOOCInit.cfg").c_str(), ios::trunc);

    if(!cfile.fail())
    {
        for(map<std::string, Loc >::iterator it = _locInit.begin();
        it != _locInit.end(); it++)
        {
            //cerr << "Writing entry for " << it->first << endl;
            cfile << it->first << " " << it->second.pointSize << " " << it->second.shaderSize  << " " << it->second.pointFunc[2] << " " << it->second.pointAlpha << " " << it->second.shaderEnabled << " ";
            for(int i = 0; i < 4; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    cfile << it->second.pos(i, j) << " ";
                }
            }
            cfile << endl;
        }
    }
    cfile.close();
}
