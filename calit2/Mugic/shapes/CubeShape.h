#ifndef _CUBESHAPE_H_
#define _CUBESHAPE_H_

/* Cube Shape for Mugic */

#include "GeometryShape.h"
#include <osg/Geometry>

class CubeShape : public GeometryShape
{
	public:
		CubeShape( std::string command = "", std::string name = "" );
		virtual ~CubeShape();
		void update( std::string );

	protected:
		CubeShape();
		void setPosition( osg::Vec3, float width, float height, float depth);
		void setColor( osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4, osg::Vec4 );
		void setTextureCoords();
		void setTextureImage(std::string);
		void update();

};

#endif
