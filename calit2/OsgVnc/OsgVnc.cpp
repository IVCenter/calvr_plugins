#include "OsgVnc.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/TextureCubeMap>
#include <osg/TexMat>
#include <osg/CullFace>
#include <osg/ImageStream>
#include <osg/io_utils>
#include <osgDB/Registry>

using namespace osg;
using namespace std;
using namespace cvr;


CVRPLUGIN(OsgVnc)

OsgVnc::OsgVnc() : FileLoadCallback("vnc")
{
}

/*
// check for intersection with vnc window and then forward correct information to the widget
// ONLY do the one the master node
bool OsgVnc::processEvent(InteractionEvent * event)
{
    // check for intersection with vncwindow
    osg::Vec3 pointerStart, pointerEnd;
    std::vector<IsectInfo> isecvec;

    pointerStart = tie->getTransform().getTrans();
    pointerEnd.set(0.0f, 10000.0f, 0.0f);
    pointerEnd = pointerEnd * tie->getTransform();

    isecvec = getObjectIntersection(cvr::PluginHelper::getScene(),
                                    pointerStart, pointerEnd);

    // If we didn't intersect, get out of here
    if (isecvec.size() == 0)
        return false;
  
    printf("Comparing geodes %d\n", isecvec.size()); 
    // check if isec geode matches a vncWindow
    for(std::vector<struct VncObject*>::iterator it = _loadedVncs.begin(); it != _loadedVncs.end(); it++)
    {
        // check for hit
        if( (*it)->window->getChildNode(0) == isecvec[0].geode )
        {
            printf("Found a hit\n");     
            return true;   
        }    
    } 

    return false;
}
*/

bool OsgVnc::loadFile(std::string filename)
{

    osgWidget::GeometryHints hints(osg::Vec3(0.0f,0.0f,0.0f),
                                   osg::Vec3(1.0f,0.0f,0.0f),
                                   osg::Vec3(0.0f,0.0f,1.0f),
                                   osg::Vec4(1.0f,1.0f,1.0f,1.0f),
                                   osgWidget::GeometryHints::RESIZE_HEIGHT_TO_MAINTAINCE_ASPECT_RATIO);

    
    std::string hostname = osgDB::getNameLessExtension(filename);
    osg::ref_ptr<osgWidget::VncClient> vncClient = new osgWidget::VncClient;
    if (vncClient->connect(hostname, hints))
    {
        struct VncObject * currentobject = new struct VncObject;
        currentobject->name = hostname;

        // add to scene object
        VncSceneObject * sot = new VncSceneObject(currentobject->name, vncClient.get(), ComController::instance()->isMaster(), false,false,false,true,true);
        PluginHelper::registerSceneObject(sot,"OsgVnc");

        // set up SceneObject
        sot->attachToScene();
        sot->setNavigationOn(true);
        sot->addMoveMenuItem();
        sot->addNavigationMenuItem();
        sot->addScaleMenuItem("Scale", 0.1, 10.0, 1.0);

        currentobject->scene = sot;
        
        // add controls
        //MenuCheckbox * mcb = new MenuCheckbox("Plane lock", false);
        //mcb->setCallback(this);
        //so->addMenuItem(mcb);
        //_planeMap[currentobject] = mb;
        
        MenuButton * mb = new MenuButton("Delete");
        mb->setCallback(this);
        sot->addMenuItem(mb);
        _deleteMap[currentobject] = mb;

        _loadedVncs.push_back(currentobject);
    }
    else
    {
	    std::cerr << "Unable to read vnc stream\n";
    }

    return true;
}

void OsgVnc::menuCallback(MenuItem* menuItem)
{

    //check map for a delete
    for(std::map<struct VncObject*, MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            for(std::vector<struct VncObject*>::iterator delit = _loadedVncs.begin(); delit != _loadedVncs.end(); delit++)
            {
                if((*delit) == it->first)
                {
		            // need to delete title SceneObject
		            if( it->first->scene )
			            delete it->first->scene;
                    it->first->scene = NULL; 
    
                    _loadedVncs.erase(delit);
                    break;
                }
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            break;
        }
    }
}

bool OsgVnc::init()
{
    std::cerr << "OsgVnc init\n";
    return true;
}

OsgVnc::~OsgVnc()
{

    while(_loadedVncs.size())
    {
         VncObject* it = _loadedVncs.at(0);
         
         // remove delete map item
         if(_deleteMap.find(it) != _deleteMap.end())
         {
            delete _deleteMap[it];
            _deleteMap.erase(it);
         }

		 if( it->scene )
		    delete it->scene;
         it->scene = NULL;

         _loadedVncs.erase(_loadedVncs.begin());
    }
}
