#ifndef _SPHERE_SHAPE_H_
#define _SPHERE_SHAPE_H_

#include "DrawableShape.h"

class SphereShape : public DrawableShape
{

	public:
		SphereShape(std::string command = "", std::string name = "");
		virtual ~SphereShape();
		void update(std::string);

	protected:
		SphereShape();
		void setPosition(osg::Vec3 position, float radius);
		void setShapeColor(osg::Vec4);
		void setShaders(std::string, std::string);
		void update();
		osg::Sphere* _sphere; //actual shape

};

#endif
