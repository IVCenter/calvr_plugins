#include "VolumeMenu.h"

using namespace cvr;

void VolumeMenu::init()
{
	scale = new MenuRangeValueCompact("Scale", 0.1, 100.0, 1.0, true);
	scale->setCallback(this);
	_scene->addMenuItem(scale);

	sampleDistance = new MenuRangeValueCompact("SampleDistance", .0001, 0.01, .001, true);
	sampleDistance->setCallback(this);
	_scene->addMenuItem(sampleDistance);


	SubMenu* contrast = new SubMenu("Contrast");
	_scene->addMenuItem(contrast);

	contrastBottom = new MenuRangeValueCompact("Contrast Bottom", 0.0, 1.0, 0.0, false);
	contrastBottom->setCallback(this);
	contrast->addItem(contrastBottom);

	contrastTop = new MenuRangeValueCompact("Contrast Top", 0.0, 1.0, 1.0, false);
	contrastTop->setCallback(this);
	contrast->addItem(contrastTop);


	SubMenu* opacity = new SubMenu("Opacity");
	_scene->addMenuItem(opacity);

	opacityMult = new MenuRangeValueCompact("Opacity Multiplier", 0.01, 10.0, 1.0, false);
	opacityMult->setCallback(this);
	opacity->addItem(opacityMult);

	opacityCenter = new MenuRangeValueCompact("Opacity Center", 0.0, 1.0, 1.0, false);
	opacityCenter->setCallback(this);
	opacity->addItem(opacityCenter);

	opacityWidth = new MenuRangeValueCompact("Opacity Width", 0.01, 1.0, 1.0, false);
	opacityWidth->setCallback(this);
	opacity->addItem(opacityWidth);


	adaptiveQuality = new MenuCheckbox("Adaptive Quality", false);
	adaptiveQuality->setCallback(this);
	_scene->addMenuItem(adaptiveQuality);

	colorFunction = new MenuList();
	std::vector<std::string> colorfunctions = std::vector<std::string>();
	colorfunctions.push_back("Default");
	colorfunctions.push_back("Rainbow");
	colorFunction->setValues(colorfunctions);

	colorFunction->setCallback(this);
	_scene->addMenuItem(colorFunction);


	SubMenu* maskoptions = new SubMenu("Mask Options", "Mask Options");
	_scene->addMenuItem(maskoptions);

	highlightColon = new MenuCheckbox("Highlight Colon", false);
	highlightColon->setCallback(this);
	maskoptions->addItem(highlightColon);

	organsOnly = new MenuCheckbox("Display organs only", false);
	organsOnly->setCallback(this);
	maskoptions->addItem(organsOnly);

}


void VolumeMenu::menuCallback(cvr::MenuItem * item)
{
	if (item == sampleDistance)
	{
		_volume->_StepSize->set(sampleDistance->getValue());
	}
	else if (item == scale)
	{
		_scene->setScale(scale->getValue());
	}
	else if (item == contrastBottom)
	{
		_volume->_computeUniforms["ContrastBottom"]->set(contrastBottom->getValue());
		_volume->setDirtyAll();
	}
	else if (item == contrastTop)
	{
		_volume->_computeUniforms["ContrastTop"]->set(contrastTop->getValue());
		_volume->setDirtyAll();
	}
	else if (item == opacityMult)
	{
		_volume->_computeUniforms["OpacityMult"]->set(opacityMult->getValue());
		_volume->setDirtyAll();
	}
	else if (item == opacityCenter)
	{
		_volume->_computeUniforms["OpacityCenter"]->set(opacityCenter->getValue());
		_volume->setDirtyAll();
	}
	else if (item == opacityWidth)
	{
		_volume->_computeUniforms["OpacityWidth"]->set(opacityWidth->getValue());
		_volume->setDirtyAll();
	}
	else if (item == highlightColon)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("COLON", highlightColon->getValue());
		_volume->setDirtyAll();
	}
	else if (item == organsOnly)
	{
		_volume->getCompute()->getOrCreateStateSet()->setDefine("ORGANS_ONLY", organsOnly->getValue());
		_volume->setDirtyAll();
	}
	else if (item == adaptiveQuality)
	{
		_volume->getDrawable()->getOrCreateStateSet()->setDefine("VR_ADAPTIVE_QUALITY", adaptiveQuality->getValue());
	}
	else if (item == colorFunction)
	{
		if (colorFunction->getIndex() == DEFAULT)
		{
			_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", osg::StateAttribute::OFF);
		}
		else if (colorFunction->getIndex() == RAINBOW)
		{
			_volume->getCompute()->getOrCreateStateSet()->setDefine("COLOR_FUNCTION", "hsv2rgb(vec3(ra.r * 0.8, 1, 1))", osg::StateAttribute::ON);
		}
		transferFunction = (ColorFunction)colorFunction->getIndex();
		_volume->setDirtyAll();
	}
}