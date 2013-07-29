#include "GraphGlobals.h"
#include "ColorGenerator.h"

#include <cvrKernel/CalVR.h>
#include <cvrConfig/ConfigManager.h>

using namespace cvr;

bool GraphGlobals::_init = false;
bool GraphGlobals::_deferUpdate = false;
osg::ref_ptr<osgText::Font> GraphGlobals::_font;
osg::Vec4 GraphGlobals::_bgColor;
osg::Vec4 GraphGlobals::_dataBGColor;
std::map<std::string,osg::Vec4> GraphGlobals::_phylumColorMap;
std::map<std::string,osg::Vec4> GraphGlobals::_patientColorMap;
osg::Vec4 GraphGlobals::_defaultPhylumColor;
osg::Vec4 GraphGlobals::_lowColor;
osg::Vec4 GraphGlobals::_normColor;
osg::Vec4 GraphGlobals::_high1Color;
osg::Vec4 GraphGlobals::_high10Color;
osg::Vec4 GraphGlobals::_high100Color;
float GraphGlobals::_pointLineScale;
float GraphGlobals::_masterPointScale;
float GraphGlobals::_masterLineScale;

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

void GraphGlobals::makeTextFit(osgText::Text * text, float maxSize, bool horizontal)
{
    checkInit();

    osg::BoundingBox bb = text->getBound();
    float width;
    if(horizontal)
    {
	width = bb.xMax() - bb.xMin();
    }
    else
    {
	width = bb.zMax() - bb.zMin();
    }

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
	if(horizontal)
	{
	    width = bb.xMax() - bb.xMin();
	}
	else
	{
	    width = bb.zMax() - bb.zMin();
	}
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

osg::Vec4 GraphGlobals::getColorLow()
{
    checkInit();
    return _lowColor;
}

osg::Vec4 GraphGlobals::getColorNormal()
{
    checkInit();
    return _normColor;
}

osg::Vec4 GraphGlobals::getColorHigh1()
{
    checkInit();
    return _high1Color;
}

osg::Vec4 GraphGlobals::getColorHigh10()
{
    checkInit();
    return _high10Color;
}

osg::Vec4 GraphGlobals::getColorHigh100()
{
    checkInit();
    return _high100Color;
}

const std::map<std::string,osg::Vec4> & GraphGlobals::getPatientColorMap()
{
    checkInit();
    return _patientColorMap;
}

bool GraphGlobals::getDeferUpdate()
{
    checkInit();
    return _deferUpdate;
}

void GraphGlobals::setDeferUpdate(bool defer)
{
    checkInit();
    _deferUpdate = defer;
}

float GraphGlobals::getPointLineScale()
{
    checkInit();
    return _pointLineScale;
}

float GraphGlobals::getMasterPointScale()
{
    checkInit();
    return _masterPointScale;
}

float GraphGlobals::getMasterLineScale()
{
    checkInit();
    return _masterLineScale;
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
    //_dataBGColor = osg::Vec4(0.7,0.7,0.7,1.0);
    _dataBGColor = osg::Vec4(0.73,0.74,0.69,1.0);

    // TODO: get real phylum color mapping
    _phylumColorMap["Bacteroidetes"] = ColorGenerator::makeColor(0,7);
    _phylumColorMap["Firmicutes"] = ColorGenerator::makeColor(1,7);
    _phylumColorMap["Verrucomicrobia"] = ColorGenerator::makeColor(2,7);
    _phylumColorMap["Proteobacteria"] = ColorGenerator::makeColor(3,7);
    _phylumColorMap["Actinobacteria"] = ColorGenerator::makeColor(4,7);
    _phylumColorMap["Fusobacteria"] = ColorGenerator::makeColor(5,7);
    _phylumColorMap["Euryarchaeota"] = ColorGenerator::makeColor(6,7);
    _defaultPhylumColor = osg::Vec4(0.4,0.4,0.4,1.0);

    std::vector<std::string> patientTypes;
    patientTypes.push_back("Smarr");
    patientTypes.push_back("Crohns");
    patientTypes.push_back("UC");
    patientTypes.push_back("Healthy");

    for(int i = 0; i < patientTypes.size(); ++i)
    {
	_patientColorMap[patientTypes[i]] = ColorGenerator::makeColor(i,patientTypes.size());
    }

    _lowColor = osg::Vec4(0.54,0.81,0.87,1.0);
    _normColor = osg::Vec4(0.63,0.67,0.40,1.0);
    _high1Color = osg::Vec4(0.86,0.61,0.0,1.0);
    _high10Color = osg::Vec4(0.86,0.31,0.0,1.0);
    _high100Color = osg::Vec4(0.71,0.18,0.37,1.0);

    _pointLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.PointLineScale",1.0);
    _masterPointScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterPointScale",1.0);
    _masterLineScale = ConfigManager::getFloat("value","Plugin.FuturePatient.MasterLineScale",1.0);
}
