#ifndef _CameraFlight_H_
#define _CameraFlight_H_

#include <list>
#include <set>
#include <vector>

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRPlugin.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/Utils>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuImage.h>
#include <menu/MenuList.h>
#include <menu/MenuText.h>
#include <menu/DialogPanel.h>
#include <menu/SubMenu.h>

#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/ShapeDrawable>

#include <string>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

enum FlightMode
{
    INSTANT,
    SATELLITE,
    AIRPLANE
};

class CameraFlight : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        CameraFlight();
        ~CameraFlight();

        bool init();

	void menuCallback(cvr::MenuItem * item);
	int getPriority() {return 52;}

        void preFrame();
	void postFrame();
	
	bool processEvent(cvr::InteractionEvent * event);
	bool processMouseEvent(cvr::MouseInteractionEvent * event);
	bool buttonEvent(int type);


	void printMat(osg::Matrix, double);
	void printVec(osg::Vec3);
	void printQuat(osg::Quat);	
	void navigate(osg::Matrix, osg::Vec3);
	void zoomIn(osg::Vec3 v, osg::Matrix mat);
	void zoomOut(osg::Vec3 v, osg::Matrix mat);
	void rotate(osg::Vec3 from, osg::Vec3 to);

    protected:
	cvr::SubMenu * _camMenu;
	cvr::SubMenu * _algoMenu;
	cvr::SubMenu * _destMenu;

	cvr::MenuCheckbox * _instant;
	cvr::MenuCheckbox * _satellite;
	cvr::MenuCheckbox * _reset;

	cvr::MenuCheckbox * _dest1;
	cvr::MenuCheckbox * _dest2;
	cvr::MenuCheckbox * _dest3;
	cvr::MenuCheckbox * _dest4;
	cvr::MenuCheckbox * _dest5;
	cvr::MenuCheckbox * _dest6;

	cvr::MenuCheckbox * activeMode;
	cvr::MenuCheckbox * destMode;
	FlightMode _flightMode;

	osg::Vec3 currentPos;
	osg::Vec3 oldPos;
	osgEarth::Map * map;
	osgEarth::Map * outputMap;
};

#endif
