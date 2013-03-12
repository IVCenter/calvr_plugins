#ifndef _CONE_SHAPE_H_
#define _CONE_SHAPE_H_

#include "DrawableShape.h"

class ConeShape : public DrawableShape
{

	public:
		ConeShape(std::string command = "", std::string name = "");
		virtual ~ConeShape();
		void update(std::string);

	protected:
		ConeShape();
		void setPosition(osg::Vec3 position, float radius, float height);
		void setShapeColor(osg::Vec4);
		void setShaders(std::string, std::string);
		void update();
		osg::Cone* _cone; //actual shape

};

#endif
