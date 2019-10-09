#include "UIExtensions.h"

VisibilityToggle::VisibilityToggle(std::string text)
	: cvr::UIToggle()
{
	_callback = NULL;

	std::string dir = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	eye = new cvr::UICheckbox(dir + "eye_closed.png", dir + "eye_open.png");
	eye->offElement->setTransparent(true);
	eye->onElement->setTransparent(true);
	eye->setAspect(osg::Vec3(1, 0, 1));
	eye->setPercentSize(osg::Vec3(0.3, 1, 0.9));
	eye->setPercentPos(osg::Vec3(0.05, 0, -0.05));
	eye->setDisplayed(0);
	eye->setAlign(cvr::UIElement::CENTER_CENTER);
	addChild(eye);

	label = new cvr::UIText(text, 50.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentSize(osg::Vec3(0.6, 1, 1));
	label->setPercentPos(osg::Vec3(0.4, 0, 0));
	addChild(label);
}

bool VisibilityToggle::onToggle()
{
	if (_on)
	{
		eye->setDisplayed(1);
	}
	else
	{
		eye->setDisplayed(0);
	}

	if (_callback)
	{
		_callback->uiCallback(this);
	}
	return true;
}