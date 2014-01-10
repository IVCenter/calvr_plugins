#include "Sensor.h"
#include <iostream>

Sensor::Sensor(bool type, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style) :  _type(type)
{
	_location.first = 0.0;
	_location.second = 0.0;
    _direction = 0.0f;
    _velocity = 0.0f;
    _temperature = 0.0f;
    _pressure = 0.0f;
    _humidity = 0.0f;
    _rotation = new FlagTransform();
    _flagText = new FlagText( 120.0 ); // set a default size
	_flagText->setFont(font.get());
	_flagText->setStyle(style.get());
    _flagcolor = new osg::Uniform("flagcolor", 0.0f);
}

FlagTransform::FlagTransform() : _changed(false)
{
	setUpdateCallback(this);	
}

void FlagTransform::operator()(osg::Node* node, osg::NodeVisitor*)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	
    FlagTransform * ft = dynamic_cast<FlagTransform *>(node);
    if (ft && _changed)
    {
    	((osg::MatrixTransform*) ft)->setMatrix(_mat);
        _changed = false;
    }
}

void FlagTransform::setMatrix(osg::Matrix mat)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_mat = mat;
    _changed = true;
}

osg::Matrix FlagTransform::getMatrix()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	return _mat;
}

FlagText::FlagText(float size) : _changed(false)
{
	// set text object params
	//osg::ref_ptr<osgText::Font> font = osgText::readFontFile("/home/pweber/development/calvr/resources/arial.ttf");

    // add text above the cylinder
    //osg::ref_ptr<osgText::Style> style = new osgText::Style;
    //style->setThicknessRatio(0.01);

	// init the text object
	//setFont(font.get());
	//setStyle(style.get());
	setCharacterSize(size);
	setFontResolution(256,256);
	setAlignment(osgText::Text::CENTER_BOTTOM);
	setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	setPosition(osg::Vec3(0.0, 0.0, 0.0));

	setUpdateCallback(this);	
}

void FlagText::update(osg::NodeVisitor*, osg::Drawable* drawable)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	
    FlagText * ft = dynamic_cast<FlagText *>(drawable);
    if (ft && _changed)
    {
        //std::cerr << "Update called: " << _text << std::endl;
       	((osgText::Text3D*) ft)->setText(_text);
        _changed = false;
    }
}

void FlagText::setText(std::string text)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_text = text;
    _changed = true;
}
