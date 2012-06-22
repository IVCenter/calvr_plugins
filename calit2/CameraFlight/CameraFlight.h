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
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuImage.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuText.h>
#include <cvrMenu/DialogPanel.h>
#include <cvrMenu/SubMenu.h>

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
	double findMaxHeight(double);
	
	void navigate(osg::Vec3);

	void normalView();

	/*For Plane Mode Only*/
	void planeView();
	void planeDir(osg::Vec3, osg::Vec3);
	void directSet(int);

	void rise(osg::Vec3, osg::Matrix);
	void zoomOut(osg::Vec3, osg::Matrix);
	void rotate(osg::Vec3, osg::Vec3);

    protected:
	cvr::SubMenu * _camMenu;
	cvr::SubMenu * _algoMenu;
	cvr::SubMenu * _destMenu;
	cvr::SubMenu * _customDestMenu;

	cvr::MenuCheckbox * _instant;
	cvr::MenuCheckbox * _satellite;
	cvr::MenuCheckbox * _airplane;
	cvr::MenuCheckbox * _reset;

	cvr::MenuCheckbox * _dest1;
	cvr::MenuCheckbox * _dest2;
	cvr::MenuCheckbox * _dest3;
	cvr::MenuCheckbox * _dest4;
	cvr::MenuCheckbox * _dest5;
	cvr::MenuCheckbox * _dest6;

	cvr::MenuCheckbox * activeMode;
	cvr::MenuCheckbox * destMode;
	cvr::MenuRangeValue * _customLat;
	cvr::MenuRangeValue * _customLon;
	cvr::MenuButton * _goButton;
	FlightMode _flightMode;

	osg::Vec3 currentPos;
	osg::Vec3 oldPos;
	osgEarth::Map * map;
	osgEarth::Map * outputMap;
};

#endif
