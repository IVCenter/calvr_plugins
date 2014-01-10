#include "ArrowNode.h"

#include <osgDB/ReadFile>
#include <osgText/Text3D>

#include <iostream>

ArrowNode::ArrowNode()
{
}

ArrowNode::~ArrowNode()
{
}


// note location needs to be computed ( location of wind data info )
ArrowNode::ArrowNode(osg::Matrix location, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style) : _speed(0), _direction(0)
{
    // set current transform
    setMatrix(location);

    // create arrow
    osg::TessellationHints* hints = new osg::TessellationHints;
    hints->setDetailRatio(0.5f);

    // create quat to deal with the rotation of the shape
    //osg::Quat(90.0, osg::Vec3(   ));

    osg::Geode* _arrowGeode = new osg::Geode();
    _arrowGeode->addDrawable(new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0.0, 0.0, -1.0), 0.5, 2.0) , hints));
    _arrowGeode->addDrawable(new osg::ShapeDrawable(new osg::Cone(osg::Vec3(0.0, 0.0, -(1.0 / 3)), 1.0, 2.0), hints));
    _arrowGeode->setUpdateCallback(this);


    // scale the arrow
    osg::MatrixTransform* scalet = new osg::MatrixTransform();
    osg::Matrix scalem;
    scalem.makeScale(0.2, 0.2, 0.2);
    scalet->setMatrix(scalem);
    scalet->addChild(_arrowGeode);
    addChild(scalet);

/*
    float height = 1000.0;

    // find height of tower and create offset for adding images
    // add auto transform for displaying images in plan of screen
    osg::AutoTransform* at = new osg::AutoTransform;
    at->setPosition(osg::Vec3(0.0, 0.0, height)); // TODO need to adjust
    at->setAutoRotateMode(osg::AutoTransform::ROTATE_TO_CAMERA);

    // TODO need to make sure it is above the flag
    // create text name of the location
    osgText::Text3D* text3D = new osgText::Text3D;
    text3D->setFont(font.get());
    text3D->setStyle(style.get());
    text3D->setCharacterSize(height / 10);
    text3D->setFontResolution(256,256);
    text3D->setAlignment(osgText::Text::CENTER_BOTTOM);
    text3D->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
    text3D->setPosition(osg::Vec3(0.0, _imageheight * 0.5f, 0.0)); // TODO need to adjust
    text3D->setText(name);

    // need to create arrow to rotate (cylinder and cone, low tesslation)
    osg::Geode* textGeode = new osg::Geode();
    _textGeode->addDrawable(text3D);

    _arrowGeode->setUpdateCallback(this);

    at->addChild(_textGeode);
    addChild(at);
    */
}

void ArrowNode::update(float direction, float speed)
{

}

void ArrowNode::operator()(osg::Node* node, osg::NodeVisitor* nv)
{
	// check if time has passed enough
	// TODO need to complete to auto change images


    // make sure image names exist
    if( _images.size() )
    {
        double timeGone = (_currentTime - _startTime);
        if( timeGone > _delay )
        {
            _index++;

            // make sure index is valid
            if( _index >= _images.size() )
                _index = 0;

            // set a new image
            setImage(_images.at(_index));

            // reset the time
            _startTime = _currentTime;

        }
    }
	// continue traversal
    osg::NodeCallback::traverse(node,nv);
}
