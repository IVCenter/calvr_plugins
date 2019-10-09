#include "UIExtensions.h"
#include "HelmsleyVolume.h"

using namespace cvr;

VisibilityToggle::VisibilityToggle(std::string text)
	: UIToggle()
{
	std::string dir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	eye = new UICheckbox(dir + "eye_closed.png", dir + "eye_open.png");
	eye->offElement->setTransparent(true);
	eye->onElement->setTransparent(true);
	eye->setAspect(osg::Vec3(1, 0, 1));
	eye->setPercentSize(osg::Vec3(0.3, 1, 0.9));
	eye->setPercentPos(osg::Vec3(0.05, 0, -0.05));
	eye->setDisplayed(0);
	eye->setAlign(UIElement::CENTER_CENTER);
	addChild(eye);

	label = new UIText(text, 50.0f, osgText::TextBase::LEFT_CENTER);
	label->setPercentSize(osg::Vec3(0.6, 1, 1));
	label->setPercentPos(osg::Vec3(0.4, 0, 0));
	addChild(label);
}

bool VisibilityToggle::onToggle()
{
	std::cerr << "TOGGLE " << _on << std::endl;
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

ToolRadial::ToolRadial()
{
}

void ToolRadial::onSelectionChange()
{
	std::cerr << "SELECTION " << _current << std::endl;
	if (_current == -1)
	{
		HelmsleyVolume::instance()->setTool(HelmsleyVolume::NONE);
	}
	else if (_current == 0)
	{
		HelmsleyVolume::instance()->setTool(HelmsleyVolume::CUTTING_PLANE);
	}
	else if (_current == 1)
	{
		HelmsleyVolume::instance()->setTool(HelmsleyVolume::MEASUREMENT_TOOL);
	}
}

ToolSelector::ToolSelector(UIList::Direction d, UIList::OverflowBehavior o)
	: UIList(d, o)
{
	_toolRadial = new ToolRadial();

	std::string dir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	UITexture* scissors = new UITexture(dir + "scissors.png");
	scissors->setTransparent(true);
	scissors->setAspect(osg::Vec3(1, 0, 1));
	scissors->setAbsolutePos(osg::Vec3(0, -0.1f, 0));

	UISwitch* indicator = new UISwitch();
	UIQuadElement* bknd = new UIQuadElement(osg::Vec4(1, 0, 0, 1));
	indicator->addChild(bknd);
	bknd = new UIQuadElement(osg::Vec4(0, 1, 0, 1));
	indicator->addChild(bknd);
	indicator->setDisplayed(0);

	UIRadialButton* btn = new UIRadialButton(_toolRadial);
	
	UIEmptyElement* listItem = new UIEmptyElement();
	listItem->addChild(scissors);
	listItem->addChild(indicator);
	listItem->addChild(btn);
	addChild(listItem);


	UITexture* measuringTape = new UITexture(dir + "ruler.png");
	measuringTape->setTransparent(true);
	measuringTape->setAspect(osg::Vec3(1, 0, 1));
	measuringTape->setAbsolutePos(osg::Vec3(0, -0.1f, 0));

	indicator = new UISwitch();
	bknd = new UIQuadElement(osg::Vec4(1, 0, 0, 1));
	indicator->addChild(bknd);
	bknd = new UIQuadElement(osg::Vec4(0, 1, 0, 1));
	indicator->addChild(bknd);
	indicator->setDisplayed(0);

	btn = new UIRadialButton(_toolRadial);

	listItem = new UIEmptyElement();
	listItem->addChild(measuringTape);
	listItem->addChild(indicator);
	listItem->addChild(btn);
	addChild(listItem);

}