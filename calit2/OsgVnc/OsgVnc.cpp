#include "OsgVnc.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
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
        currentobject->vnc = vncClient.get();

        // add to scene object
        SceneObject * so = new SceneObject(currentobject->name,false,false,false,true,true);
        PluginHelper::registerSceneObject(so,"OsgVnc");
        so->addChild(vncClient.get());
        so->attachToScene();
        so->setNavigationOn(true);
        so->addMoveMenuItem();
        so->addNavigationMenuItem();
        so->addScaleMenuItem("Scale", 0.1, 10.0, 1.0);

        currentobject->scene = so;
        
        // add controls
        //MenuCheckbox * mcb = new MenuCheckbox("Plane lock", false);
        //mcb->setCallback(this);
        //so->addMenuItem(mcb);
        //_planeMap[currentobject] = mb;
        
        
        MenuButton * mb = new MenuButton("Delete");
        mb->setCallback(this);
        so->addMenuItem(mb);
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
                    // close the stream first
                    it->first->vnc->close();

		            // need to delete the SceneObject
		            if( it->first->scene )
			            delete it->first->scene;

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

OsgVnc::~OsgVnc() // TODO destroy all connections cleanly
{

    while(_loadedVncs.size())
    {
         VncObject* it = _loadedVncs.at(0);
         
         // close the stream first
         it->vnc->close();

         // remove delete map item
         if(_deleteMap.find(it) != _deleteMap.end())
         {
            delete _deleteMap[it];
            _deleteMap.erase(it);
         }

		 // need to delete the SceneObject
		 if( it->scene )
		    delete it->scene;

          _loadedVncs.erase(_loadedVncs.begin());
    }
}
