#include "TextShape.h"

#include <osg/Geometry>
#include <osg/Material>

#include <string>
#include <vector>
#include <iostream>

ThreadMap< std::string, osgText::Font* > * TextShape::_fontMap = NULL;

TextShape::TextShape(std::string command, std::string name) 
{
    // check for changed values
    createParameter("pos", new Vec3Type());
    createParameter("color", new Vec4Type());
    createParameter("bcolor", new Vec4Type());
    createParameter("enableOutline", new BoolType());
    createParameter("size", new FloatType());
    createParameter("label", new StringType());
    createParameter("font", new StringType());

    _type = SimpleShape::TEXT;

    BasicShape::setName(name);
    
    // default color is white for font
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));

    // set a default background color
    setBackdropColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));

    setCharacterSize(10.0f);
    setFontResolution(40,40);

    osgText::Font* font = NULL;

    // check if fontMap has been constucted
    if ( _fontMap == NULL )
        _fontMap = new ThreadMap< std::string, osgText::Font* >();

    // check if font exists in map else load it and add it
    if( ! _fontMap->get(std::string("/usr/share/fonts/liberation/LiberationSans-Regular.ttf"), font) )
    {
        font = osgText::readFontFile("/usr/share/fonts/liberation/LiberationSans-Regular.ttf");
        _fontMap->add(std::string("/usr/share/fonts/liberation/LiberationSans-Regular.ttf"), font);
    }

    setFont(font);

    if( Globals::G_ROTAXIS )
        setAxisAlignment(osgText::TextBase::XY_PLANE);
    else
        setAxisAlignment(osgText::TextBase::XZ_PLANE);

    // off by default
    setBackdropType(osgText::Text::NONE);

    osg::StateSet* state = osgText::Text::getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //osg::Material* mat = new osg::Material();
    //mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //state->setAttributeAndModes(mat, osg::StateAttribute::ON);
   
    setUpdateCallback(new TextUpdateCallback());
    
    update(command);
}

osg::Geode* TextShape::getParent()
{
    return osgText::Text::getParent(0)->asGeode();
}

osg::Drawable* TextShape::asDrawable()
{
    return dynamic_cast<osg::Drawable*>(this);
}

TextShape::~TextShape()
{
}

void TextShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    // TODO deal with triples for values e.g. color/backdrop
    setParameter(command, "pos");
    setParameter(command, "color");
    setParameter(command, "bcolor");
    setParameter(command, "size");
    setParameter(command, "label");
    setParameter(command, "enableOutline");
    setParameter(command, "font");
}

void TextShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p = getPosition();
    osg::Vec4 c = getColor();
    osg::Vec4 bc = getBackdropColor();
    float size = getCharacterHeight();
    std::string text = getText().createUTF8EncodedString();

    p = getParameter("pos")->asVec3Type()->getValue();
    c = getParameter("color")->asVec4Type()->getValue();
    bc = getParameter("bcolor")->asVec4Type()->getValue();
    size = getParameter("size")->asFloatType()->getValue();
    text = getParameter("label")->asStringType()->getValue();
    bool enableOutline = getParameter("enableOutline")->asBoolType()->getValue();

    // TODO background color
    
    // TODO not sure if want to make it so the font gets added all the time hmmm
    //setFont(font);

    // check if outline should be enabled   
    if( enableOutline )
    {
        setBackdropType(osgText::Text::OUTLINE);
        setColor(c);
        setBackdropColor(bc);
    }
    else
    {
        setBackdropType(osgText::Text::NONE);
        setColor(c);
    }

    setText(text);
    setCharacterSize(size);
    setPosition(p);
    
    dirtyBound();

    if(c[3] != 1.0)
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

	// reset flag
    _dirty = false;
}
