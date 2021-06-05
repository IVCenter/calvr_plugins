#ifndef Selection3D_H
#define Selection3D_H


#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include <osg/CullFace>

#include <osgText/Text>

#include <cvrKernel/SceneObject.h>

#include <iostream>



class Selection3DTool : public cvr::SceneObject
{
public:
	Selection3DTool(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
 	}

	void setStart(osg::Vec3 v);
	void setEnd(osg::Vec3 v);
 
	float getLength();
	void activate();
	void deactivate();

protected:
	void init();
	void update();

 	osg::ref_ptr<osg::MatrixTransform> _ruler;

	osg::Vec3 _start;
	osg::Vec3 _end;
	osg::Uniform* _ustart;
	osg::Uniform* _uend;
	osg::StateSet* _stateset;
	bool isBackFace = false;
};

#endif