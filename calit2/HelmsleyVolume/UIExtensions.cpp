#include "UIExtensions.h"
#include "HelmsleyVolume.h"
#include "cvrKernel/NodeMask.h"

using namespace cvr;

osg::Program* ColorPickerSaturationValue::_svprogram = nullptr;


#pragma region VisibilityToggle
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

#pragma endregion

#pragma region CallbackToggle
bool CallbackToggle::onToggle()
{
	if (_callback)
	{
		_callback->uiCallback(this);
	}
	return true;
}
#pragma endregion

#pragma region CallbackRadial
void CallbackRadial::onSelectionChange()
{
	if (_callback)
	{
		_callback->uiCallback(this);
	}
}
#pragma endregion

#pragma region ToolRadialButton
ToolRadialButton::ToolRadialButton(UIRadial* parent, std::string iconpath)
	: UIRadialButton(parent)
{
	_icon = new UITexture(iconpath);
	_icon->setTransparent(true);
	_icon->setColor(osg::Vec4(0, 0, 0, 1));
	_icon->setPercentSize(osg::Vec3(0.8, 1, 0.8));
	_icon->setPercentPos(osg::Vec3(0.1, 0, -0.1));

	_quad = new UIQuadElement(osg::Vec4(0.95, 0.95, 0.95, 1));
	_quad->setRounding(0.0f, 0.05f);
	_quad->setTransparent(true);
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
#pragma endregion

#pragma region ToolToggle
ToolToggle::ToolToggle(std::string iconpath)
	: CallbackToggle()
{
	_icon = new UITexture(iconpath);
	_icon->setTransparent(true);
	_icon->setColor(osg::Vec4(0, 0, 0, 1));
	_icon->setPercentSize(osg::Vec3(0.8, 1, 0.8));
	_icon->setPercentPos(osg::Vec3(0.1, 0, -0.1));

	_quad = new UIQuadElement(osg::Vec4(0.95, 0.95, 0.95, 1));
	_quad->setRounding(0.0f, 0.05f);
	_quad->setTransparent(true);
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
#pragma endregion

#pragma region CallbackSlider
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
#pragma endregion

#pragma region ShaderQuad
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
#pragma endregion

#pragma region PlanePointer
void PlanePointer::setPointer(float x, float y)
{
	osg::Vec2 pos = osg::Vec2(x, y);
	if (_pointer != pos)
	{
		_pointer = pos;
		_dirty = true;
	}
}

void PlanePointer::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();
	//_geode = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_intersect);
	//_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);

	osg::Geometry* drawable = UIUtil::makeQuad(1, 1, osg::Vec4(1, 1, 1, 1), osg::Vec3(0, 0, 0));
	//_geode->addDrawable(drawable);
	//should quadelement have an intersectable?
	_intersect->addDrawable(drawable);

	updateGeometry();

}

void PlanePointer::updateGeometry()
{
	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);
}

bool PlanePointer::processEvent(InteractionEvent* event)
{
	TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();
	if (tie && tie->getButton() == _button)
	{
		if (tie->getInteraction() == BUTTON_DOWN)
		{
			_held = true;
			return true;
		}
		else if (tie->getInteraction() == BUTTON_DRAG && _held)
		{
			osg::MatrixList ltw = _intersect->getWorldMatrices();
			osg::Matrix m = ltw[0];//osg::Matrix::identity();
			//for (int i = 0; i < ltw.size(); ++i)
			//{
			//	m.postMult(ltw[i]);
			//}
			osg::Matrix mi = osg::Matrix::inverse(m);

			osg::Vec4d l4 = osg::Vec4(0, 1, 0, 0) * TrackingManager::instance()->getHandMat(tie->getHand());
			osg::Vec3 l = osg::Vec3(l4.x(), l4.y(), l4.z());

			osg::Vec4d l04 = osg::Vec4(0, 0, 0, 1) * TrackingManager::instance()->getHandMat(tie->getHand());
			osg::Vec3 l0 = osg::Vec3(l04.x(), l04.y(), l04.z());

			osg::Vec4d n4 = osg::Vec4(0, 1, 0, 0) * m;
			osg::Vec3 n = osg::Vec3(n4.x(), n4.y(), n4.z());

			osg::Vec4d p04 = osg::Vec4(0, 0, 0, 1) * m;
			osg::Vec3 p0 = osg::Vec3(p04.x(), p04.y(), p04.z());


			osg::Vec3 p = l0 + l * (((p0 - l0) * n) / (l * n));

			osg::Vec4 pl = osg::Vec4(p.x(), p.y(), p.z(), 1) * mi;

			setPointer(pl.x(), pl.y());
			//_percent = pl.x(); // _lastHitPoint.x();
			//_dirty = true;
			//std::cerr << "<" << _lastHitPoint.x() << ", " << _lastHitPoint.y() << ", " << _lastHitPoint.z() << ">" << std::endl;
			return onPosChange();
		}
		else if (tie->getInteraction() == BUTTON_UP)
		{
			_held = false;
			return false;
		}
	}

	return false;
}
#pragma endregion

#pragma region ColorPickerSV
ColorPickerSaturationValue::ColorPickerSaturationValue()
	: PlanePointer()
{
	setPointer(1, 1);
	_sv = osg::Vec2(1, 1);
	_shader = new ShaderQuad();
	_shader->setProgram(getOrLoadProgram());
	_shader->addUniform("Hue");
	addChild(_shader);
}

bool ColorPickerSaturationValue::onPosChange()
{
	_sv = _pointer;
	_indicator->setPercentPos(osg::Vec3(_pointer.x(), _pointer.y(), 0));
	if (_callback)
	{
		_callback->uiCallback(this);
	}
}

void ColorPickerSaturationValue::setHue(float hue)
{
	if (_hue != hue)
	{
		_hue = hue;
		_shader->getUniform("Hue")->set(_hue);
	}
}

osg::Program* ColorPickerSaturationValue::getOrLoadProgram()
{
	if (!_svprogram)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("colorpickerSV.frag");
		_svprogram = new osg::Program;
		_svprogram->setName("ColorpickerSV");
		_svprogram->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_svprogram->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _svprogram;
}
#pragma endregion

#pragma region ColorPickerH
ColorPickerHue::ColorPickerHue()
	: PlanePointer()
{
	setPointer(0, 0);
	_hue = 0;
	_shader = new ShaderQuad();
	_shader->setProgram(getOrLoadProgram());
	_shader->addUniform("SV");
	addChild(_shader);
}

bool ColorPickerHue::onPosChange()
{
	_hue = _pointer.y();
	_indicator->setPercentPos(osg::Vec3(0, _pointer.y(), 0));
	if (_callback)
	{
		_callback->uiCallback(this);
	}
}

void ColorPickerHue::setSV(osg::Vec2 SV)
{
	if (_sv != SV)
	{
		_sv = SV;
		_shader->getUniform("SV")->set(_sv);
	}
}

osg::Program* ColorPickerHue::getOrLoadProgram()
{
	if (!_hueprogram)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("colorpickerH.frag");
		_hueprogram = new osg::Program;
		_hueprogram->setName("ColorpickerSV");
		_hueprogram->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_hueprogram->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _hueprogram;
}
#pragma endregion

#pragma region ColorPicker
ColorPicker::ColorPicker() :
	UIElement()
{
	color = osg::Vec3(0, 1, 1);

	_bknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	addChild(_bknd);
	
	float border = 25;

	_hue = new ColorPickerHue();
	_hue->setAbsolutePos(osg::Vec3(border, border, 0.1f));
	_hue->setSize(osg::Vec3(0.8f, 1.0f, 1.0f), osg::Vec3(-border*2, -border*2, 0.0f));
	_bknd->addChild(_hue);

	_sv = new ColorPickerSaturationValue();
	_sv->setPos(osg::Vec3(0.8f, 0.0f, 0.0f), osg::Vec3(border, border, 0.1f));
	_sv->setSize(osg::Vec3(0.2f, 1.0f, 1.0f), osg::Vec3(-border*2, -border*2, 0.0f));
	_bknd->addChild(_sv);
}

void ColorPicker::uiCallback(UICallbackCaller* ui)
{
	if (ui == _hue)
	{
		float h = _hue->getHue();
		if (color.x() != h)
		{
			color.x() = h;
			if (_callback)
			{
				_callback->uiCallback(this);
			}
		}
		_sv->setHue(h);
	}
	else if(ui == _sv)
	{
		osg::Vec2 sv = _sv->getSV();
		if (color.y() != sv.x() || color.z() != sv.y())
		{
			color.y() = sv.x();
			color.z() = sv.y();
			if (_callback)
			{
				_callback->uiCallback(this);
			}
		}

		_hue->setSV(sv);
	}
}
#pragma endregion