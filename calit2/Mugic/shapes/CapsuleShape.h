#ifndef _CAPSULE_SHAPE_H_
#define _CAPSULE_SHAPE_H_

#include "DrawableShape.h"

class CapsuleShape : public DrawableShape
{

	public:
		CapsuleShape(std::string command = "", std::string name = "");
		virtual ~CapsuleShape();
		void update(std::string);

	protected:
		CapsuleShape();
		void setPosition(osg::Vec3 position, float radius, float height);
		void setShapeColor(osg::Vec4);
		void setShaders(std::string, std::string);
		void update();
		osg::Capsule* _capsule; //actual shape

};

#endif
