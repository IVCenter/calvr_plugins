#include "TextShape.h"

#include <osg/Geometry>

#include <string>
#include <vector>
#include <iostream>

TextShape::TextShape(std::string command, std::string name) 
{
    _type = SimpleShape::TEXT;

    BasicShape::setName(name);
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
    setCharacterSize(10.0f);
    setFontResolution(40,40);
    setAxisAlignment(osgText::TextBase::XZ_PLANE);
    setFont("/home/pweber/extern_libs/OpenSceneGraph-Data/fonts/arial.ttf");
    //setText("HELLO");
    
    update(command);
}

TextShape::~TextShape()
{
}

void TextShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x");
    addParameter(command, "y");
    addParameter(command, "z");
    addParameter(command, "r");
    addParameter(command, "g");
    addParameter(command, "b");
    addParameter(command, "a");
    addParameter(command, "size");
    addParameter(command, "label");
}

void TextShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p = getPosition();
    osg::Vec4 c = getColor();
    float size = getCharacterHeight();
    std::string text = getText().createUTF8EncodedString();

    setParameter("x", p.x()); 
    setParameter("y", p.y()); 
    setParameter("z", p.z()); 
    setParameter("r", c.r()); 
    setParameter("g", c.b()); 
    setParameter("b", c.g()); 
    setParameter("a", c.a());
    setParameter("size", size);
    setParameter("label", text);
    
    std::cerr << "text value " << text << std::endl;
    
    setText(text);
    setCharacterSize(size);
    setPosition(p);
    setColor(c);

	// reset flag
    _dirty = false;
}

void TextShape::drawImplementation(osg::RenderInfo &renderInfo) const
{
    osgText::Text::drawImplementation(renderInfo);    
}
