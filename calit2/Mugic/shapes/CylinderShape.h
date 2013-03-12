#ifndef _CYLINDER_SHAPE_H_
#define _CYLINDER_SHAPE_H_

#include "DrawableShape.h"

class CylinderShape : public DrawableShape
{

	public:
		CylinderShape(std::string command = "", std::string name = "");
		virtual ~CylinderShape();
		void update(std::string);

	protected:
		CylinderShape();
		void setPosition(osg::Vec3 position, float radius, float height);
		void setShapeColor(osg::Vec4);
		void setShaders(std::string, std::string);
		void update();
		osg::Cylinder* _cylinder; //actual shape

};

#endif
