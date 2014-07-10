#include "VirvoVolume.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/Intersection.h>
#include <iostream>

#include <osg/Node>
#include <osg/Geometry>
#include <osg/Notify>
#include <osg/Geode>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/Endian>
#include <osg/ShapeDrawable>
#include <osg/PolygonMode>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Point>
#include <osgUtil/Simplifier>

#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include "VirvoDrawable.h"

#include <string.h>
#include <fstream>

using namespace std;
using namespace cvr;

// uid generator
int VirvoVolume::id;

CVRPLUGIN(VirvoVolume)



VirvoVolume::VirvoVolume() : FileLoadCallback("xvf")
{
}


struct VirvoVolume::volumeinfo* VirvoVolume::loadXVF(std::string filename)
{
    struct volumeinfo* volinfo = NULL;

    vvVolDesc* vol = new vvVolDesc(filename.c_str());

	vvFileIO vvIO;
	vvFileIO::ErrorType type = vvIO.loadVolumeData(vol);
	if(type != vvFileIO::OK)
	{
	    std::cerr << "Error reading XVF file\n";
	    delete vol;
	    return volinfo;
	}

	// create a new volume
	volinfo = new struct volumeinfo;
	volinfo->name = osgDB::getSimpleFileName(filename);
	volinfo->desc = vol;
    volinfo->id = id++;

    // create copy of default tranfer function
    volinfo->defaultTransferFunc = new vvTransFunc;
    vvTransFunc::copy(&volinfo->defaultTransferFunc->_widgets,  &volinfo->desc->tf._widgets);
    volinfo->defaultTransferFunc->setDiscreteColors(volinfo->desc->tf.getDiscreteColors());
   
    return volinfo;
}


bool VirvoVolume::loadFile(std::string filename)
{
    
	volumeinfo* info = loadXVF(filename);
   
	// failed to load volume
	if( info == NULL )
		return false;

	info->volume = new osg::Geode();
	info->drawable = new VirvoDrawable();
	info->drawable->enableFlatDisplay(false);
   	info->drawable->setROIPosition(osg::Vec3(0.,0.,0.));
	info->drawable->setVolumeDescription( info->desc );
	info->volume->addDrawable(info->drawable);

    // create scene object
    SceneObject* so = new SceneObject(info->name, false, false, false, true, true);
	so->addChild(info->volume);
    so->setBoundsCalcMode(SceneObject::MANUAL);
    so->setBoundingBox(info->drawable->getBound());
    PluginHelper::registerSceneObject(so,"VirvoVolume");

    double xMin, xMax, yMin, yMax, zMin, zMax;
    xMin = info->drawable->getBound().xMin();
    xMax = info->drawable->getBound().xMax();
    yMin = info->drawable->getBound().yMin();
    yMax = info->drawable->getBound().yMax();
    zMin = info->drawable->getBound().zMin();
    zMax = info->drawable->getBound().zMax();

    // create volume clipping plane
    info->clippingPlane = new SceneObject("ClippingPlane", false, true, false, false, true);
   
    // manually set the bounding box
    osg::BoundingBox bound(xMin, yMin, -0.1, xMax, yMax, 0.1);
    info->clippingPlane->setBoundsCalcMode(SceneObject::MANUAL);
    info->clippingPlane->setBoundingBox(bound);

    // add menu item to enable and disable clipping plane
    MenuCheckbox* cpcb = new MenuCheckbox("Clip Plane", false);
    cpcb->setCallback(this);
    so->addMenuItem(cpcb);
    _clipplaneMap[so] = cpcb;
  
    // create submenu for adjusting transfer function
    SubMenu* transfer = new SubMenu("Transfer Functions");
    so->addMenuItem(transfer);

    MenuCheckbox* mtdcb = new MenuCheckbox("Default", true);
    mtdcb->setCallback(this);
    transfer->addItem(mtdcb);
    _transferDefaultMap[so] = mtdcb;

    MenuCheckbox* mctbb = new MenuCheckbox("Bright", false);
    mctbb->setCallback(this);
    transfer->addItem(mctbb);
    _transferBrightMap[so] =  mctbb;

    MenuCheckbox* mcthb = new MenuCheckbox("Hue", false);
    mcthb->setCallback(this);
    transfer->addItem(mcthb);
    _transferHueMap[so] = mcthb;

    MenuCheckbox* mctgb = new MenuCheckbox("Gray", false);
    mctgb->setCallback(this);
    transfer->addItem(mctgb);
    _transferGrayMap[so] = mctgb;

    MenuRangeValue* mrvp = new MenuRangeValue("Position", 0.0, 1.0, 0.5);
    mrvp->setCallback(this);
    transfer->addItem(mrvp);
    _transferPositionMap[so] = mrvp;

    MenuRangeValue* mrvbw = new MenuRangeValue("Base Width", 0.0, 2.0, 1.0);
    mrvbw->setCallback(this);
    transfer->addItem(mrvbw);
    _transferBaseWidthMap[so] = mrvbw;

    // try and set default positions
    if( info->defaultTransferFunc->_widgets.size() != 0 )
    {
        // saved transfer functions are generally Pyr types
        vvTFPyramid* pyr = dynamic_cast<vvTFPyramid*> (info->defaultTransferFunc->_widgets[0]);
        if( pyr )
        {
            mrvp->setValue(pyr->_pos[0]);
            mrvbw->setValue(pyr->_bottom[0]);
            std::cerr << " Center " << pyr->_pos[0] << " width " << pyr->_bottom[0] << std::endl;
        }
    }

    // add animation menu if applicable
    if( info->drawable->getNumFrames() > 1 )
    {
        SubMenu* animation = new SubMenu("Animation");
        so->addMenuItem(animation);

        MenuCheckbox* mcpb = new MenuCheckbox("Play", false);
        mcpb->setCallback(this);
        animation->addItem(mcpb);
        _playMap[so] = mcpb;

        MenuRangeValue* mrvsb = new MenuRangeValue("Speed", 0.0, 10.0, 0.4);
        mrvsb->setCallback(this);
        animation->addItem(mrvsb);
        _speedMap[so] = mrvsb;
        
        MenuRangeValue* mrvfb = new MenuRangeValue("Frame", 0.0, info->drawable->getNumFrames(), 0.0);
        mrvfb->setCallback(this);
        animation->addItem(mrvfb);
        _frameMap[so] = mrvfb;
    }

    // add save position button
    MenuButton* mb = new MenuButton("Save position");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _saveMap[so] = mb;

    // add delete button
    mb = new MenuButton("Delete");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _deleteMap[so] = mb;

    so->attachToScene();
    so->setNavigationOn(true);
    so->addMoveMenuItem();
    so->addNavigationMenuItem();
    so->addScaleMenuItem("Scale", 0.1, 10.0, 1.0);

    // check if there exists a preset configuration
    bool nav;
    nav = so->getNavigationOn();
    so->setNavigationOn(false);

    if(_locInit.find(info->name) != _locInit.end())
    {
         so->setTransform(_locInit[info->name].pos);
         so->setScale(so->getScale());
         info->clippingPlane->setTransform(_locInit[info->name].clip);
    }

    so->setNavigationOn(nav);

    _volumeMap[so] = info;

    return true;
}

void VirvoVolume::menuCallback(MenuItem* menuItem)
{

	//check for main menu selections
    std::map< cvr::MenuItem* , std::string>::iterator it = _menuFileMap.find(menuItem);
    if( it != _menuFileMap.end() )
    {
        loadFile(it->second);
        return;
    }

    // check for remove all button
    if( menuItem == _removeButton )
    {
        removeAll();
		return;
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _clipplaneMap.begin(); it != _clipplaneMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // see if clip plane is enabled or disabled
            if(it->second->getValue())
            {
                it->first->addChild(info->clippingPlane);
                info->drawable->setClipping(true);
            }
            else
            {
                it->first->removeChild(info->clippingPlane);
                info->drawable->setClipping(false);
            }

        }
    }

    //compute and set transfer function
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferDefaultMap.begin(); it != _transferDefaultMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // only used when clicked
            if(it->second->getValue())
            {
                // copy default transfer function values 
                vvTransFunc transfunc;
                vvTransFunc::copy(&transfunc._widgets, &info->defaultTransferFunc->_widgets);
                transfunc.setDiscreteColors(info->defaultTransferFunc->getDiscreteColors());

                // adjust the function
                adjustTransferFunction(transfunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() ); 
                
                // set the new function
                info->drawable->setTransferFunction( &transfunc );

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }


    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferGrayMap.begin(); it != _transferGrayMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                vvTransFunc transfunc;

                // adjust the function
                adjustTransferFunction(transfunc, 2, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );

                // set the new function
                info->drawable->setTransferFunction( &transfunc );
                                                               
                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferBrightMap.begin(); it != _transferBrightMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                vvTransFunc transfunc;

                // adjust the function
                adjustTransferFunction(transfunc, 0, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );

                // set the new function
                info->drawable->setTransferFunction( &transfunc );

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferHueMap.begin(); it != _transferHueMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                vvTransFunc transfunc;

                // adjust the function
                adjustTransferFunction(transfunc, 1, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );

                // set the new function
                info->drawable->setTransferFunction( &transfunc );

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
                mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    // check slider position movement for updates
    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _transferPositionMap.begin(); it != _transferPositionMap.end(); it++)
    {
        // make sure one of the modes is set
        if(menuItem == it->second && (_transferDefaultMap[it->first]->getValue() || _transferGrayMap[it->first]->getValue() || _transferBrightMap[it->first]->getValue() || _transferHueMap[it->first]->getValue()))
        {
           volumeinfo* info = _volumeMap[it->first];
           
           // find out what color index to use // TODO use default colormap
           int colorIndex = -1;
           if( _transferBrightMap[it->first]->getValue() )
               colorIndex = 0;
           if( _transferHueMap[it->first]->getValue() )
               colorIndex = 1;
           else if( _transferGrayMap[it->first]->getValue() )
               colorIndex = 2;

           vvTransFunc transfunc;

           // first value is color palete, second position, third base width
           if( colorIndex == -1 )
		   {
                vvTransFunc::copy(&transfunc._widgets, &info->defaultTransferFunc->_widgets);
                transfunc.setDiscreteColors(info->defaultTransferFunc->getDiscreteColors());
                adjustTransferFunction(transfunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );
           }
		   else
		   {
                // adjust the function
                adjustTransferFunction(transfunc, colorIndex, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );
           }

           // set the new function
           info->drawable->setTransferFunction( &transfunc );
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _transferBaseWidthMap.begin(); it != _transferBaseWidthMap.end(); it++)
    {
        // make sure one of the modes is set
        if(menuItem == it->second && (_transferDefaultMap[it->first]->getValue() || _transferGrayMap[it->first]->getValue() || _transferBrightMap[it->first]->getValue() || _transferHueMap[it->first]->getValue()))
        {
           volumeinfo* info = _volumeMap[it->first];

           // find out what color index to use //TODO use default colormap
           int colorIndex = -1;
           if( _transferBrightMap[it->first]->getValue() )
               colorIndex = 0;
           if( _transferHueMap[it->first]->getValue() )
               colorIndex = 1;
           else if( _transferGrayMap[it->first]->getValue() )
               colorIndex = 2;

           // copy default transfer function values 
           vvTransFunc transfunc;
           
           // first value is color palete, second position, third base width
		   if( colorIndex == -1 )
		   {
                vvTransFunc::copy(&transfunc._widgets, &info->defaultTransferFunc->_widgets);
                transfunc.setDiscreteColors(info->defaultTransferFunc->getDiscreteColors());

                // adjust the function
                adjustTransferFunction(transfunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() ); 
           }
		   else
		   {
                // adjust the function
                adjustTransferFunction(transfunc, colorIndex, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue() );
		   }

           // set the new function
           info->drawable->setTransferFunction( &transfunc );
        }
    }

    // animation playback 
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _playMap.begin(); it != _playMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(cvr::ComController::instance()->isMaster())
            {
                volumeinfo* info = _volumeMap[it->first];

				//TODO not an image sequence
                
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _speedMap.begin(); it != _speedMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(cvr::ComController::instance()->isMaster())
            {
                volumeinfo* info = _volumeMap[it->first];

				//TODO not an image sequence
               
            }
            return;
        }
    }
    
    for(std::map<SceneObject*,MenuButton*>::iterator it = _saveMap.begin(); it != _saveMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
           
            bool nav;
            nav = it->first->getNavigationOn();
            it->first->setNavigationOn(false);

            loc temploc;
            temploc.scale = 1.0;
            temploc.pos = it->first->getTransform();
            temploc.clip = info->clippingPlane->getTransform(); 
            _locInit[it->first->getName()] = temploc;
            it->first->setNavigationOn(nav);

            writeConfigFile();
            return;
        }
    }

    for(std::map<SceneObject*,MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            deleteVolume(it->first);
            return;
        }
    }

}

void VirvoVolume::deleteVolume(cvr::SceneObject* vol)
{
    if(_clipplaneMap.find(vol) != _clipplaneMap.end())
    {
        delete _clipplaneMap[vol];
        _clipplaneMap.erase(vol);
    }

    if(_transferPositionMap.find(vol) != _transferPositionMap.end())
    {
        delete _transferPositionMap[vol];
        _transferPositionMap.erase(vol);
    }

    if(_transferPositionMap.find(vol) != _transferPositionMap.end())
    {
        delete _transferPositionMap[vol];
        _transferPositionMap.erase(vol);
    }

    if(_transferBaseWidthMap.find(vol) != _transferBaseWidthMap.end())
    {
        delete _transferBaseWidthMap[vol];
        _transferBaseWidthMap.erase(vol);
    }

    if(_transferBrightMap.find(vol) != _transferBrightMap.end())
    {
        delete _transferBrightMap[vol];
        _transferBrightMap.erase(vol);
    }

    if(_transferHueMap.find(vol) != _transferHueMap.end())
    {
        delete _transferHueMap[vol];
        _transferHueMap.erase(vol);
    }

    if(_transferGrayMap.find(vol) != _transferGrayMap.end())
    {
        delete _transferGrayMap[vol];
        _transferGrayMap.erase(vol);
    }
    
    if(_saveMap.find(vol) != _saveMap.end())
    {
        delete _saveMap[vol];
        _saveMap.erase(vol);
    }

    if(_deleteMap.find(vol) != _deleteMap.end())
    {
        delete _deleteMap[vol];
        _deleteMap.erase(vol);
    }

    if(_playMap.find(vol) != _playMap.end())
    {
        delete _playMap[vol];
        _playMap.erase(vol);
    }
    
    if(_speedMap.find(vol) != _speedMap.end())
    {
        delete _speedMap[vol];
        _speedMap.erase(vol);
    }
    
    if(_frameMap.find(vol) != _frameMap.end())
    {
        delete _frameMap[vol];
        _frameMap.erase(vol);
    }

    if(_volumeMap.find(vol) != _volumeMap.end())
    {
        delete _volumeMap[vol];
        _volumeMap.erase(vol);

        // TODO fix delete (might need to remove clipping plane SceneObject first)
        delete vol;
        vol = NULL;
    }
}

void VirvoVolume::adjustTransferFunction(vvTransFunc& tf, float position, float baseWidth)
{
    // if min and max are not 0.0 then add new pyramid widget
    if( position != 0.0 && baseWidth != 0.0)
    {
        // remove old pyramid Widgets
        tf.deleteWidgets(vvTFWidget::TF_PYRAMID);
        tf.deleteWidgets(vvTFWidget::TF_BELL);
        tf.deleteWidgets(vvTFWidget::TF_CUSTOM);
        tf.deleteWidgets(vvTFWidget::TF_SKIP);
        tf._widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, position, baseWidth, 0.0f));
    }
}

void VirvoVolume::adjustTransferFunction(vvTransFunc& tf, int colorTable, float position, float baseWidth)
{
    if( colorTable < 0 || colorTable > 2)
        return;

    // remove previous data
    tf.deleteWidgets(vvTFWidget::TF_PYRAMID);
    tf.deleteWidgets(vvTFWidget::TF_BELL);
    tf.deleteWidgets(vvTFWidget::TF_CUSTOM);
    tf.deleteWidgets(vvTFWidget::TF_SKIP);

    tf.setDefaultColors(colorTable, 0.0, 1.0);
    tf._widgets.push_back(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, position, baseWidth, 0.0f));
}

void VirvoVolume::preFrame()
{
    // exit preframe if no volumes
    if( ! _volumeMap.size() )
        return;
	
	osg::Matrix viewerWorld = PluginHelper::getHeadMat();
    osg::Vec3 viewDirWorld(viewerWorld(1, 0), viewerWorld(1, 1), viewerWorld(1, 2));

    // iterate through the volume map and set the viewer and object direction for renderering
    for(std::map<cvr::SceneObject*, volumeinfo*>::iterator it = _volumeMap.begin(); it != _volumeMap.end(); ++it)
    {
        osg::Vec3 bbCenterObj = it->first->getOrComputeBoundingBox().center();
        
        osg::Matrix bbCenterMatrix;
        bbCenterMatrix.setTrans(bbCenterObj);
        osg::Matrix obj2world = it->first->getObjectToWorldMatrix();

        osg::Vec3 bbCenterWorld = bbCenterObj * obj2world;
        osg::Matrix bbCenterWorld2obj = osg::Matrix::inverse(bbCenterMatrix * obj2world);
        
        osg::Vec3 objDirWorld = bbCenterWorld - viewerWorld.getTrans();

        osg::Vec3 viewDirObj = viewDirWorld * bbCenterWorld2obj;
        osg::Vec3 objDirObj = objDirWorld * bbCenterWorld2obj;
        objDirObj.normalize();

        volumeinfo* info = _volumeMap[it->first];
        info->drawable->setViewDirection(viewDirObj);
        info->drawable->setObjectDirection(objDirObj);
        info->drawable->setQuality(_quality->getValue());
    }


    // check if clipping plane is enabled if so set location and direction for applicable volume
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _clipplaneMap.begin(); it != _clipplaneMap.end(); it++)
    {
        volumeinfo* info = _volumeMap[it->first];
     
        if( it->second->getValue() )
        {
            osg::Matrixd transform = info->clippingPlane->getTransform();

            // set normal and point
            osg::Vec3d normal = transform.getRotate() * osg::Vec3(0.0, 0.0, 1.0);
            normal.normalize();
            info->drawable->setClipPoint(transform.getTrans());
            info->drawable->setClipDirection(normal);
        }
     }

/*
    // create and id lookup map
    std::map<int, animationinfo > timeLookup;

    // send all animation data also apply clipping plane to all volumes
    std::map<cvr::SceneObject*,volumeinfo*>::iterator it = _volumeMap.begin();
    for(; it != _volumeMap.end(); ++it)
    {
        animationinfo dataPacket;

        volumeinfo* info = _volumeMap[it->first];

        // current it supports a single clipping plane so only support the first clipping plane
        // check if a clip plane has been enabled
        if( PluginHelper::getObjectsRoot()->getNumClipPlanes() > 0 )
        {
            osg::Vec4d p = PluginHelper::getObjectsRoot()->getClipPlane(0)->getClipPlane();
            
            // convert to point and normal
            osg::Vec3d normal(p[0], p[1], p[2]);
            osg::Vec3d point( -p[3] / p[0], -p[3] / p[1], -p[3] / p[2]);

            //std::cerr << "Object Point " << point[0] << " " << point[1] << " " << point[2] << " normal " << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;

            // convert the point and normal to volume space and then apply to volume
            osg::Vec3d vpoint = point * PluginHelper::getObjectToWorldTransform() * it->first->getWorldToObjectMatrix();
            osg::Vec3d vnormal = (point + normal) * PluginHelper::getObjectToWorldTransform() * it->first->getWorldToObjectMatrix();
            vnormal = vnormal - vpoint;
            vnormal.normalize();
          
            if( vnormal.valid() && vpoint.valid() )
            { 
                // set clipping plane in volume and enable
                info->drawable->setClipDirection(vnormal);
                info->drawable->setClipPoint(vpoint);
                info->drawable->setClipping(true);
            }
            
            //std::cerr << "Scene Object Point " << vpoint[0] << " " << vpoint[1] << " " << vpoint[2] << " normal " << vnormal[0] << " " << vnormal[1] << " " << vnormal[2] << std::endl;
        }
        else // disable clipping plane
        {
            info->drawable->setClipping(false);
        }

        cvrImageSequence* sequence = dynamic_cast<cvrImageSequence*> (it->second->image.get());
        if( sequence )
        {
            if(cvr::ComController::instance()->isMaster())
            {
                dataPacket.id = it->second->id;
                dataPacket.time = sequence->getCurrentTime();
                dataPacket.frame = sequence->getFrame(dataPacket.time); 
                //std::cerr << "Current time " << dataPacket.time << "  and frame " << dataPacket.frame << std::endl;
                cvr::ComController::instance()->sendSlaves(&dataPacket,sizeof(dataPacket));
            }
            else
            {
                cvr::ComController::instance()->readMaster(&dataPacket,sizeof(dataPacket));
            }
            timeLookup[dataPacket.id] = dataPacket;
        }

    }

    // update slaves
    if(!cvr::ComController::instance()->isMaster())
    {
        // update all the seek locations for the image sequences
        for(it = _volumeMap.begin(); it != _volumeMap.end(); ++it)
        {
            volumeinfo* info = _volumeMap[it->first];
            
            osg::ImageSequence* sequence = dynamic_cast<osg::ImageSequence*> (it->second->image.get());
            if( sequence )
            {
                if(timeLookup.find(it->second->id) != timeLookup.end())
                {
                    animationinfo animinfo = timeLookup[it->second->id];
                    sequence->seek(animinfo.time);
                }

            }

        }
    }

    // update slider
    for(it = _volumeMap.begin(); it != _volumeMap.end(); ++it)
    {
        // update slider
        if(_frameMap.find(it->first) != _frameMap.end())
        {
            animationinfo animinfo = timeLookup[it->second->id];
            _frameMap[it->first]->setValue(animinfo.frame);
        }
    }
*/
}

bool VirvoVolume::init()
{
    std::cerr << "VirvoVolume init\n";
    //osg::setNotifyLevel( osg::INFO );

    // create default menu
    _volumeMenu = new SubMenu("VirvoVolume", "VirvoVolume");
    _volumeMenu->setCallback(this);

    _filesMenu = new SubMenu("Files","Files");
    _filesMenu->setCallback(this);
    _volumeMenu->addItem(_filesMenu);
    
    _quality = new MenuRangeValue("Quality", 0.0, 10.0, 1.0);
    _quality->setCallback(this);
    _volumeMenu->addItem(_quality);
    
    _removeButton = new MenuButton("Remove All");
    _removeButton->setCallback(this);
    _volumeMenu->addItem(_removeButton);

    // read in configurations
    _configPath = ConfigManager::getEntry("Plugin.VirvoVolume.ConfigDir");

    ifstream cfile;
    cfile.open((_configPath + "/VirvoInit.cfg").c_str(), ios::in);

    if(!cfile.fail())
    {
        string line;
        while(!cfile.eof())
        {
            osg::Matrix p;
            osg::Matrix c;
            float scale;
            char name[150];

            // read in name
            cfile >> name;
            if(cfile.eof())
            {
                break;
            }

            // read in scale
			cfile >> scale;

            // read in position
			for(int i = 0; i < 4; i++)
        	{
        		for(int j = 0; j < 4; j++)
        		{
            		cfile >> p(i, j);
        		}
        	}
			
            // read in clip plane 
            for(int i = 0; i < 4; i++)
        	{
        		for(int j = 0; j < 4; j++)
        		{
            		cfile >> c(i, j);
        		}
            }
            //_locInit[string(name)] = pair<float, osg::Matrix>(scale, m);
            
            loc temploc;
            temploc.scale = scale;
            temploc.pos = p;
            temploc.clip = c;
            _locInit[string(name)] = temploc;
        }
    }
    cfile.close();

    // read in configuartion files
    vector<string> list;

    string configBase = "Plugin.VirvoVolume.Files";

    ConfigManager::getChildren(configBase,list);

    for(int i = 0; i < list.size(); i++)
    {
        MenuButton * button = new MenuButton(list[i]);
        button->setCallback(this);

        // add mapping
        _menuFileMap[button] = ConfigManager::getEntry("value",configBase + "." + list[i],"");

        // add button
        _filesMenu->addItem(button);
    }

    // add menu
    cvr:MenuSystem::instance()->addMenuItem(_volumeMenu);


    return true;
}

void VirvoVolume::removeAll()
{
    std::map<cvr::SceneObject*,volumeinfo*>::iterator it;

    while( (it = _volumeMap.begin())  != _volumeMap.end() )
    {
        deleteVolume(it->first);
    }
}

void VirvoVolume::writeConfigFile()
{
    // only write on head node
    if(cvr::ComController::instance()->isMaster())
    {

        ofstream cfile;
        cfile.open((_configPath + "/Init.cfg").c_str(), ios::trunc);

        if(!cfile.fail())
        {
    	    for(map<std::string, loc >::iterator it = _locInit.begin();
        	    it != _locInit.end(); it++)
    	    {
        	    //cerr << "Writing entry for " << it->first << endl;
        	    cfile << it->first << " " << it->second.scale << " ";
        	    
                // save volume position
                for(int i = 0; i < 4; i++)
        	    {
        		    for(int j = 0; j < 4; j++)
        		    {
            		    cfile << it->second.pos(i, j) << " ";
        		    }
        	    }
        	   
                // save clip position 
                for(int i = 0; i < 4; i++)
        	    {
        		    for(int j = 0; j < 4; j++)
        		    {
            		    cfile << it->second.clip(i, j) << " ";
        		    }
                }
        	    cfile << endl;
    	    }
        }
        cfile.close();
    }
}



VirvoVolume::~VirvoVolume()
{
   printf("Called VirvoVolume destructor\n");
}
