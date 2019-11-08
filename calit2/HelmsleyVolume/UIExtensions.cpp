#include "UIExtensions.h"
#include "HelmsleyVolume.h"

using namespace cvr;

VisibilityToggle::VisibilityToggle(std::string text)
	: UIToggle()
{
	std::string dir = ConfigManager::getEntry("Plugin.HelmsleyVolume.ImageDir");
	eye = new UICheckbox(dir + "eye_closed.png", dir + "eye_open.png");
	eye->offElement->setTransparent(true);
	eye->offElement->setColor(osg::Vec4(0,0,0,1));
	eye->onElement->setTransparent(true);
	eye->onElement->setColor(osg::Vec4(0, 0, 0, 1));
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

bool CallbackToggle::onToggle()
{
	if (_callback)
	{
		_callback->uiCallback(this);
	}
	return true;
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

ToolRadialButton::ToolRadialButton(UIRadial* parent, std::string iconpath)
	: UIRadialButton(parent)
{
	_icon = new UITexture(iconpath);
	_icon->setTransparent(true);
	_icon->setColor(osg::Vec4(0, 0, 0, 1));
	_icon->setPercentSize(osg::Vec3(0.8, 1, 0.8));
	_icon->setPercentPos(osg::Vec3(0.1, 0, -0.1));

	_quad = new UIQuadElement(osg::Vec4(0.95, 0.95, 0.95, 1));
	_quad->addChild(_icon);
	addChild(_quad);
}

void ToolRadialButton::processHover(bool enter)
{
	if (enter)
	{
		_quad->setColor(osg::Vec4(0.8, 0.8, 0.8, 1));
	}
	else
	{
		_quad->setColor(osg::Vec4(0.95, 0.95, 0.95, 1.0));
	}
}

ToolToggle::ToolToggle(std::string iconpath)
	: CallbackToggle()
{
	_icon = new UITexture(iconpath);
	_icon->setTransparent(true);
	_icon->setColor(osg::Vec4(0, 0, 0, 1));
	_icon->setPercentSize(osg::Vec3(0.8, 1, 0.8));
	_icon->setPercentPos(osg::Vec3(0.1, 0, -0.1));

	_quad = new UIQuadElement(osg::Vec4(0.95, 0.95, 0.95, 1));
	_quad->addChild(_icon);
	addChild(_quad);
}

void ToolToggle::processHover(bool enter)
{
	if (enter)
	{
		_quad->setColor(osg::Vec4(0.8, 0.8, 0.8, 1));
	}
	else
	{
		_quad->setColor(osg::Vec4(0.95, 0.95, 0.95, 1.0));
	}
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