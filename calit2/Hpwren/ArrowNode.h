#ifndef _ARROW_NODE_
#define _ARROW_NODE_

#include <vector>
#include <string>
#include <algorithm>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Geode>
#include <osg/Shape>
#include <osgText/Text>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/AutoTransform>
#include <osg/NodeVisitor>

// callback class for showing animating air quality
class ArrowNode : public osg::MatrixTransform, public osg::NodeCallback
{  
    public:
        ArrowNode(osg::Matrix location, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style);
        virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);
		void update(float direction, float speed); // do inside calvr preframe

    protected:
        ArrowNode();
        ~ArrowNode();

		// image geode
        osg::Geode* _arrowGeode;

        float _speed;
        float _direction;
};

#endif
