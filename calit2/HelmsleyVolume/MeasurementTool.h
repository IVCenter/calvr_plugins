#ifndef MEASUREMENT_TOOL_H
#define MEASUREMENT_TOOL_H


#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include <osgText/Text>

#include <cvrKernel/SceneObject.h>

#include <iostream>


class MeasurementTool : public cvr::SceneObject
{
public:
	MeasurementTool(std::string name, bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds = false)
		: SceneObject(name, navigation, movable, clip, contextMenu, showBounds)
	{
		init();
	}

	void setStart(osg::Vec3 v);
	void setEnd(osg::Vec3 v);
	void setText(std::string s);

	float getLength();
	void activate();
	void deactivate();

protected:
	void init();
	void update();

	osg::ref_ptr<osgText::Text> _text;
	osg::ref_ptr<osg::MatrixTransform> _ruler;

	osg::Vec3 _start;
	osg::Vec3 _end;
	osg::Uniform* _ustart;
	osg::Uniform* _uend;
};

#endif