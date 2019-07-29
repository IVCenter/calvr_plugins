#ifndef _DRAWABLE_SHAPE_H_
#define _DRAWABLE_SHAPE_H_

#include "BasicShape.h"
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Drawable>
#include <osg/MatrixTransform>

class DrawableShape : public BasicShape, public osg::ShapeDrawable
{

	public:
		virtual void update(std::string command) = 0;
		osg::MatrixTransform* getMatrixParent();
		osg::Geode* getParent();
		osg::Drawable* asDrawable();

		struct DrawableUpdateCallback : public osg::Drawable::UpdateCallback
		{
			virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
			{
				DrawableShape* shape = dynamic_cast<DrawableShape*> (drawable);
				if( shape )
					shape->update();
			}
		};

	protected:
		DrawableShape();
		virtual ~DrawableShape();
		float _radius;
		float _height;
		osg::Vec3 _center;
		osg::Vec4 _color;
		std::string _vertex_shader;
		std::string _fragment_shader;
		
		virtual void update() = 0;

};

#endif
