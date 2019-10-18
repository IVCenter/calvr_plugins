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

void CallbackRadial::onSelectionChange()
{
	if (_callback)
	{
		_callback->uiCallback(this);
	}
}

ToolSelector::ToolSelector(UIList::Direction d, UIList::OverflowBehavior o)
	: UIList(d, o)
{
	_toolRadial = new CallbackRadial();

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


void CallbackSlider::setMax(float max)
{
	_max = max;
}

void CallbackSlider::setMin(float min)
{
	_min = min;
}

float CallbackSlider::getMax()
{
	return _max;
}

float CallbackSlider::getMin()
{
	return _min;
}

bool CallbackSlider::onPercentChange()
{
	if (_callback)
	{
		_callback->uiCallback(this);
		return true;
	}
	return false;
}

void ShaderQuad::updateGeometry()
{
	UIQuadElement::updateGeometry();

	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);
	}
}

void ShaderQuad::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void ShaderQuad::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* ShaderQuad::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void ShaderQuad::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
}