#ifndef _TouchDesigner_
#define _TouchDesigner_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrKernel/PluginHelper.h>

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osgAnimation/EaseMotion>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "shapes/BasicShape.h"
#include "shapes/CircleShape.h"
#include "shapes/TriangleShape.h"
#include "shapes/RectShape.h"
#include "util/ShapeHelper.h"
#include "socket/SocketThread.h"

#include<osg/ShapeDrawable>
#include<osg/Shape>
#include<osg/Texture2D>
#include<osg/TexEnv>
#include<osg/StateSet>
#include<osg/Image>

#include <math.h>
#include <vector>
#include <string>

#include <osgDB/WriteFile>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>

#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
using namespace osg;
using namespace osgDB;


class CVRSocket;

class TouchDesigner : public cvr::MenuCallback, public cvr::CVRPlugin
{
public:        
	TouchDesigner();
	virtual ~TouchDesigner();

	bool init();
	void preFrame();
	void menuCallback(cvr::MenuItem*);

protected:
	cvr::SubMenu* _menu;
	cvr::MenuButton* _receiveButton;	
	void receiveGeometry();
	std::string _port;


	// returns a random number btwn 0 and 1
	double random();
	// returns a random number btwn min and max
	int random(int,int);

	ShapeHelper * sh;
	SocketThread * st;
	Node * prevNode;
};

#endif
