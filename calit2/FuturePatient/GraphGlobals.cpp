#include "GraphGlobals.h"
#include "ColorGenerator.h"

#include <cvrKernel/CalVR.h>

using namespace cvr;

bool GraphGlobals::_init = false;
osg::ref_ptr<osgText::Font> GraphGlobals::_font;
osg::Vec4 GraphGlobals::_bgColor;
osg::Vec4 GraphGlobals::_dataBGColor;
std::map<std::string,osg::Vec4> GraphGlobals::_phylumColorMap;
osg::Vec4 GraphGlobals::_defaultPhylumColor;

osgText::Text * GraphGlobals::makeText(std::string text, osg::Vec4 color)
{
    checkInit();

    osgText::Text * textNode = new osgText::Text();
    textNode->setCharacterSize(1.0);
    textNode->setAlignment(osgText::Text::CENTER_CENTER);
    textNode->setColor(color);
    textNode->setBackdropColor(osg::Vec4(0,0,0,0));
    textNode->setAxisAlignment(osgText::Text::XZ_PLANE);
    textNode->setText(text);
    textNode->setFontResolution(128,128);
    if(_font)
    {
	textNode->setFont(_font);
    }
    return textNode;
}

void GraphGlobals::makeTextFit(osgText::Text * text, float maxSize)
{
    checkInit();

    osg::BoundingBox bb = text->getBound();
    float width = bb.xMax() - bb.xMin();
    if(width <= maxSize)
    {
	return;
    }

    std::string str = text->getText().createUTF8EncodedString();
    if(!str.length())
    {
	return;
    }

    while(str.length() > 1)
    {
	str = str.substr(0,str.length()-1);
	text->setText(str + "..");
	bb = text->getBound();
	width = bb.xMax() - bb.xMin();
	if(width <= maxSize)
	{
	    return;
	}
    }

    str += ".";
    text->setText(str);
}

const osg::Vec4 & GraphGlobals::getBackgroundColor()
{
    checkInit();
    return _bgColor;
}

const osg::Vec4 & GraphGlobals::getDataBackgroundColor()
{
    checkInit();
    return _dataBGColor;
}

const std::map<std::string,osg::Vec4> & GraphGlobals::getPhylumColorMap()
{
    checkInit();
    return _phylumColorMap;
}

osg::Vec4 GraphGlobals::getDefaultPhylumColor()
{
    checkInit();
    return _defaultPhylumColor;
}

void GraphGlobals::checkInit()
{
    if(!_init)
    {
	init();
	_init = true;
    }
}

void GraphGlobals::init()
{
    _font = osgText::readFontFile(CalVR::instance()->getResourceDir() + "/resources/arial.ttf");
    if(_font)
    {
	_font->setTextureSizeHint(2048,2048);
    }

    _bgColor = osg::Vec4(0.9,0.9,0.9,1.0);
    _dataBGColor = osg::Vec4(0.7,0.7,0.7,1.0);

    // TODO: get real phylum color mapping
    _phylumColorMap["Bacteroidetes"] = ColorGenerator::makeColor(0,7);
    _phylumColorMap["Firmicutes"] = ColorGenerator::makeColor(1,7);
    _phylumColorMap["Verrucomicrobia"] = ColorGenerator::makeColor(2,7);
    _phylumColorMap["Proteobacteria"] = ColorGenerator::makeColor(3,7);
    _phylumColorMap["Actinobacteria"] = ColorGenerator::makeColor(4,7);
    _phylumColorMap["Fusobacteria"] = ColorGenerator::makeColor(5,7);
    _phylumColorMap["Euryarchaeota"] = ColorGenerator::makeColor(6,7);
    _defaultPhylumColor = osg::Vec4(0.4,0.4,0.4,1.0);
}
