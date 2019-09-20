#ifndef MEASUREMENT_TOOL_H
#define MEASUREMENT_TOOL_H


#include <osg/Group>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <iostream>


class MeasurementTool : public osg::Group
{
public:
	MeasurementTool();
	virtual ~MeasurementTool();
	MeasurementTool(const MeasurementTool &, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY);

	virtual Object* cloneType() const { return NULL; }
	virtual Object* clone(const osg::CopyOp& copyop) const { return new MeasurementTool(*this, copyop); }
	virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const MeasurementTool*>(obj) != NULL; }
	virtual const char* libraryName() const { return "HelmsleyVolume"; }
	virtual const char* className() const { return "MeasurementTool"; }



	void setStart(osg::Vec3 v);
	void setEnd(osg::Vec3 v);
	void setText(std::string s);

	float getLength();

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