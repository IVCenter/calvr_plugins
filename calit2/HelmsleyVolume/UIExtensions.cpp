#include "UIExtensions.h"
#include "HelmsleyVolume.h"
#include "cvrKernel/NodeMask.h"
#include <osg/BlendFunc>
#include <osg/PolygonMode>
#include <algorithm>

using namespace cvr;

osg::Program* ColorPickerSaturationValue::_svprogram = nullptr;
osg::Program* ColorPickerHue::_hueprogram = nullptr;
osg::Program* HistQuad::_histprogram = nullptr;
osg::Program* Tent::_triangleProg = nullptr;
osg::Program* CurvedQuad::_curvedQuadProg = nullptr;
osg::Program* TriangleButton::_triangleButtonProg = nullptr;
osg::Program* Dial::_dialProg = nullptr;
osg::Program* Line::_lineProg = nullptr;
osg::Program* MarchingCubesRender::_mcProg = nullptr;
osg::Program* ColorSlider::_colorSlideProg = nullptr;


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
	//std::cerr << "TOGGLE " << _on << std::endl;
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
	//_icon->setColor(osg::Vec4(0, 0, 0, 1));
	_icon->setPercentSize(osg::Vec3(0.8, 1, 0.8));
	_icon->setPercentPos(osg::Vec3(0.1, 0, -0.1));

	_quad = new UIQuadElement(osg::Vec4(1.00, 1.00, 1.00, 1));
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

void ToolToggle::setColor(osg::Vec4 color)
{
	_icon->setColor(osg::Vec4(0, 0, 0, 1));
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
ShaderQuad::~ShaderQuad() {
	
}

void ShaderQuad::updateGeometry()
{
	UIQuadElement::updateGeometry();

	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);
	}
}

template <typename T>
void ShaderQuad::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
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

			setPointer(pl.x(), pl.z());
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
	_shader->getUniform("Hue")->set(0.0f);
	addChild(_shader);

	_indicator = new UIQuadElement();
	_indicator->setSize(osg::Vec3(0, 1, 0), osg::Vec3(20, 0, 20));
	_indicator->setAbsolutePos(osg::Vec3(-10, -0.2f, 10));
	_indicator->setPercentPos(osg::Vec3(1.0, 0.0, 0.0));
	addChild(_indicator);
}

bool ColorPickerSaturationValue::onPosChange()
{
	_sv = _pointer;
	_sv.x() = std::min(std::max(0.0f, _sv.x()), 1.0f);
	_sv.y() = 1.0f - std::min(std::max(0.0f, -_sv.y()), 1.0f);
	_indicator->setPercentPos(osg::Vec3(_sv.x(), 0, -1.0f +_sv.y()));
	if (_callback)
	{
		_callback->uiCallback(this);
	}
	return true;
}

void ColorPickerSaturationValue::setHue(float hue)
{
	if (_hue != hue)
	{
		_hue = hue;
		_shader->getUniform("Hue")->set(_hue);
	}
}

void ColorPickerSaturationValue::set_indicator() {
	_indicator->setPercentPos(osg::Vec3(_sv.x(), 0, -1.0f + _sv.y()));
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
	_shader->addUniform("SV", osg::Vec2(1.0f, 1.0f));
	addChild(_shader);

	_indicator = new UIQuadElement();
	_indicator->setSize(osg::Vec3(1, 1, 0), osg::Vec3(0, 0, 20));
	_indicator->setAbsolutePos(osg::Vec3(0, -0.2f, 10));
	addChild(_indicator);
}

bool ColorPickerHue::onPosChange()
{
	_hue = 1.0f - std::min(std::max(0.0f, -_pointer.y()), 1.0f);
	_indicator->setPercentPos(osg::Vec3(0, 0, -1.0f + _hue));

	//changebox
	

	//
	if (_callback)
	{
		_callback->uiCallback(this);
	}
	return true;
}

void ColorPickerHue::setSV(osg::Vec2 SV)
{
	if (_sv != SV)
	{
		_sv = SV;
		_shader->getUniform("SV")->set(_sv);
	}
}

void ColorPickerHue::set_indicator() {
	_hue = 1.0f - std::min(std::max(0.0f, -_pointer.y()), 1.0f);
	_indicator->setPercentPos(osg::Vec3(0, 0, -1.0f + _hue));
}

osg::Program* ColorPickerHue::getOrLoadProgram()
{
	if (!_hueprogram)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("colorpickerH.frag");
		_hueprogram = new osg::Program;
		_hueprogram->setName("ColorpickerH");
		_hueprogram->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_hueprogram->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _hueprogram;
}
#pragma endregion

void TentWindow::setDialList(UIList* list) {
	list->setPercentPos(osg::Vec3(0, 0, -.50));
	list->setPercentSize(osg::Vec3(1, 1, .5));
	list->setAbsoluteSpacing(1);
}

TentWindow::TentWindow(TentWindowOnly* tWOnly) :
	UIElement()
{
	_tWOnly = tWOnly;
	_bknd = new UIQuadElement(osg::Vec4(.04, .25, .4, 1.0));
	_bknd->setTransparent(false);
	_bknd->setBorderSize(0.01);
	addChild(_bknd);
	_bknd->setPercentSize(osg::Vec3(1, 1, 2.0));
	
	initDials();
	

	/*_tents = std::make_unique<std::vector<Tent*>>();*/

	


	/*_tents.push_back(_tent);*/
	_tentIndex = 0;



	UIList* list = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	list->setPercentPos(osg::Vec3(0, 0, 0));
	list->setPercentSize(osg::Vec3(1, 1, 1));
	list->setAbsoluteSpacing(1);
	_bknd->addChild(list);

	
	
	


	

	_centerPos = new CallbackSlider();
	_centerPos->setPercentSize(osg::Vec3(1.8, 1, 2));
	_centerPos->setPercentPos(osg::Vec3(-.8, 0, 1));
	_centerPos->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_centerPos->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_centerPos->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_centerPos->handle->setColor(osg::Vec4(0.82, .25, .11, 0.0));
	_centerPos->filled->setColor(osg::Vec4(0.94, .44, .11, 0.16));
	_centerPos->setMax(1.0f);
	_centerPos->setMin(0.001f);
	_centerPos->setCallback(this);
	_centerPos->setPercent(.5);
	

	_bottomWidth = new CallbackSlider();
	_bottomWidth->setPercentSize(osg::Vec3(1.8, 1, 2));
	_bottomWidth->setPercentPos(osg::Vec3(-.8, 0, 1));
	_bottomWidth->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_bottomWidth->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_bottomWidth->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_bottomWidth->handle->setColor(osg::Vec4(0.82, .25, .11, 0.0));
	_bottomWidth->filled->setColor(osg::Vec4(0.94, .44, .11, 0.16));
	_bottomWidth->setMax(1.0f);
	_bottomWidth->setMin(0.001f);
	_bottomWidth->setCallback(this);
	_bottomWidth->setPercent(1.0);


	_topWidth = new CallbackSlider();
	_topWidth->setPercentSize(osg::Vec3(1.8, 1, 2));
	_topWidth->setPercentPos(osg::Vec3(-.8, 0, 1));
	_topWidth->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_topWidth->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_topWidth->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_topWidth->handle->setColor(osg::Vec4(0.82, .25, .11, 0.0));
	_topWidth->filled->setColor(osg::Vec4(0.94, .44, .11, 0.16));
	_topWidth->setMax(1.0f);
	_topWidth->setMin(0.001f);
	_topWidth->setCallback(this);
	_topWidth->setPercent(1.0);


	_height = new CallbackSlider();
	_height->setPercentSize(osg::Vec3(1.8, 1, 2));
	_height->setPercentPos(osg::Vec3(-.8, 0, 1));
	_height->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_height->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_height->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_height->handle->setColor(osg::Vec4(0.82, .25, .11, 0.0));
	_height->filled->setColor(osg::Vec4(0.94, .44, .11, 0.16));
	_height->setMax(1.0f);
	_height->setMin(0.001f);
	_height->setCallback(this);
	_height->setPercent(1.0);


	_bottom = new CallbackSlider();
	_bottom->setPercentSize(osg::Vec3(1.8, 1, 2));
	_bottom->setPercentPos(osg::Vec3(-.8, 0, 1));
	_bottom->handle->setAbsoluteSize(osg::Vec3(20, 0, 0));
	_bottom->handle->setAbsolutePos(osg::Vec3(-10, -0.2f, 0));
	_bottom->handle->setPercentSize(osg::Vec3(0, 1, 1));
	_bottom->handle->setColor(osg::Vec4(0.82, .25, .11, 0.0));
	_bottom->filled->setColor(osg::Vec4(0.94, .44, .11, 0.16));
	_bottom->setMax(1.0f);
	_bottom->setMin(0.001f);
	_bottom->setCallback(this);
	_bottom->setPercent(0.001);


	UIText* cLabel = new UIText("Center", 45.0f, osgText::TextBase::LEFT_CENTER);
	cLabel->setColor(osg::Vec4(.66, .84, .96, 1.0));
	cLabel->setPercentPos(osg::Vec3(.05, 0, 0));
	cVLabel = new UIText("0.50", 40.0f, osgText::TextBase::RIGHT_CENTER);
	cVLabel->setPercentPos(osg::Vec3(-.1, 0, 0));
	cVLabel->setColor(osg::Vec4(0.90, .90, .90, 1.0));

	UIText* bLabel = new UIText("Bottom Width", 45.0f, osgText::TextBase::LEFT_CENTER);
	bLabel->setColor(osg::Vec4(.66, .84, .96, 1.0));
	bLabel->setPercentPos(osg::Vec3(.05, 0, 0));
	bVLabel = new UIText("1.0", 45.0f, osgText::TextBase::RIGHT_CENTER);
	bVLabel->setPercentPos(osg::Vec3(-.05, 0, 0));
	bVLabel->setColor(osg::Vec4(0.90, .90, .90, 1.0));

	UIText* tLabel = new UIText("Top Width", 45.0f, osgText::TextBase::LEFT_CENTER);
	tLabel->setColor(osg::Vec4(.66, .84, .96, 1.0));
	tLabel->setPercentPos(osg::Vec3(.05, 0, 0));
	tVLabel = new UIText("1.00", 45.0f, osgText::TextBase::RIGHT_CENTER);
	tVLabel->setPercentPos(osg::Vec3(-.05, 0, 0));
	tVLabel->setColor(osg::Vec4(0.90, .90, .90, 1.0));

	UIText* hLabel = new UIText("Overall", 45.0f, osgText::TextBase::LEFT_CENTER);
	hLabel->setColor(osg::Vec4(.66, .84, .96, 1.0));
	hLabel->setPercentPos(osg::Vec3(.05, 0, 0));
	hVLabel = new UIText("1.00", 45.0f, osgText::TextBase::RIGHT_CENTER);
	hVLabel->setPercentPos(osg::Vec3(-.05, 0, 0));
	hVLabel->setColor(osg::Vec4(0.90, .90, .90, 1.0));


	UIText* lLabel = new UIText("Lowest", 45.0f, osgText::TextBase::LEFT_CENTER);
	lLabel->setColor(osg::Vec4(.66, .84, .96, 1.0));
	lLabel->setPercentPos(osg::Vec3(.05, 0, 0));
	lVLabel = new UIText("0.00", 45.0f, osgText::TextBase::RIGHT_CENTER);
	lVLabel->setPercentPos(osg::Vec3(-.05, 0, 0));
	lVLabel->setColor(osg::Vec4(0.90, .90, .90, 1.0));

	
	UIList* list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	setDialList(list2);	//makes new list and sets variables
	list2->addChild(_dialCenter);
	list2->addChild(_centerPos);
	UIList* valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	valueList->addChild(cLabel);
	valueList->addChild(cVLabel);
	list->addChild(valueList);
	list->addChild(list2);

	list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	setDialList(list2);	
	list2->addChild(_dialBW);
	list2->addChild(_bottomWidth);
	valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	valueList->addChild(bLabel);
	valueList->addChild(bVLabel);
	list->addChild(valueList);
	list->addChild(list2);

	list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	setDialList(list2);
	list2->addChild(_dialTW);
	list2->addChild(_topWidth);
	valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	valueList->addChild(tLabel);
	valueList->addChild(tVLabel);
	list->addChild(valueList);
	list->addChild(list2);

	list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	setDialList(list2);
	list2->addChild(_dialHeight);
	list2->addChild(_height);
	valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	valueList->addChild(hLabel);
	valueList->addChild(hVLabel);
	list->addChild(valueList);
	list->addChild(list2);

	list2 = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	setDialList(list2);
	list2->addChild(_dialBottom);
	list2->addChild(_bottom);
	valueList = new UIList(UIList::LEFT_TO_RIGHT, UIList::CUT);
	valueList->addChild(lLabel);
	valueList->addChild(lVLabel);
	list->addChild(valueList);
	list->addChild(list2);
	
	
}

void TentWindow::uiCallback(UICallbackCaller* ui) {
	if (ui == _bottomWidth) {
		_volume->_computeUniforms["OpacityWidth"]->setElement(_tentIndex, _bottomWidth->getAdjustedValue());
		float width = _tWOnly->_tents.at(_tentIndex)->changeBottomVertices(_bottomWidth->getAdjustedValue());
		_volume->_computeUniforms["OpacityTopWidth"]->setElement(_tentIndex, width);
		_topWidth->setPercent(width);
		bVLabel->setText(std::to_string(_bottomWidth->getAdjustedValue()).substr(0, 4));
		
	}
	if (ui == _topWidth) {
		float width = _tWOnly->_tents.at(_tentIndex)->changeTopVertices(_topWidth->getAdjustedValue());
		_volume->_computeUniforms["OpacityTopWidth"]->setElement(_tentIndex, width);
		_topWidth->setPercent(width);
		tVLabel->setText(std::to_string(_topWidth->getAdjustedValue()).substr(0, 4));
	}
	if (ui == _centerPos) {
		_tWOnly->_tents.at(_tentIndex)->setPercentPos(osg::Vec3(_centerPos->getAdjustedValue(), _tWOnly->_tents.at(_tentIndex)->getPercentPos().y(), 0.0));
		_tWOnly->_tents.at(_tentIndex)->setCenter(_centerPos->getAdjustedValue());
		_volume->_computeUniforms["OpacityCenter"]->setElement(_tentIndex, _centerPos->getAdjustedValue());
		cVLabel->setText(std::to_string(_centerPos->getAdjustedValue()).substr(0, 4));
 	}
	if (ui == _height) {
		if (_tWOnly->_tents.at(_tentIndex)->getSavedHeight() == 0.0) {
			float height = _tWOnly->_tents.at(_tentIndex)->changeHeight(_height->getAdjustedValue());
			_volume->_computeUniforms["OpacityMult"]->setElement(_tentIndex, _height->getAdjustedValue());
			_volume->_computeUniforms["Lowest"]->setElement(_tentIndex, height);
			_bottom->setPercent(height);
			hVLabel->setText(std::to_string(_height->getAdjustedValue()).substr(0, 4));
		}
	}
	if (ui == _bottom) {
		float height = _tWOnly->_tents.at(_tentIndex)->changeBottomHeight(_bottom->getAdjustedValue());
		_volume->_computeUniforms["Lowest"]->setElement(_tentIndex, height);
		_bottom->setPercent(height);
		lVLabel->setText(std::to_string(_bottom->getAdjustedValue()).substr(0, 4));
	}

	if (ui == _dialCenter)
	{
		float dialDiff = _dialCenter->getValue();
		if (dialDiff != 0.0) {
			_centerPos->setPercent(std::min(_centerPos->getAdjustedValue() + (dialDiff/2.0f), 1.0f));

			_tWOnly->_tents.at(_tentIndex)->setPercentPos(osg::Vec3(_centerPos->getAdjustedValue(), _tWOnly->_tents.at(_tentIndex)->getPercentPos().y(), 0.0));
			_tWOnly->_tents.at(_tentIndex)->setCenter(_centerPos->getAdjustedValue());
			_volume->_computeUniforms["OpacityCenter"]->setElement(_tentIndex, _centerPos->getAdjustedValue());
			cVLabel->setText(std::to_string(_centerPos->getAdjustedValue()).substr(0, 4));
		}
	}
	if (ui == _dialHeight)
	{
		float dialDiff = _dialHeight->getValue();
		if (dialDiff != 0.0) {
			if (_tWOnly->_tents.at(_tentIndex)->getSavedHeight() == 0.0) {
				_height->setPercent(std::min(_height->getAdjustedValue() + (dialDiff / 2.0f), 1.0f));

				float height = _tWOnly->_tents.at(_tentIndex)->changeHeight(_height->getAdjustedValue());
				_volume->_computeUniforms["OpacityMult"]->setElement(_tentIndex, _height->getAdjustedValue());
				_volume->_computeUniforms["Lowest"]->setElement(_tentIndex, height);
				_bottom->setPercent(height);
				hVLabel->setText(std::to_string(_height->getAdjustedValue()).substr(0, 4));
			}
		}
	}
	if (ui == _dialBW)
	{
		float dialDiff = _dialBW->getValue();
		if (dialDiff != 0.0) {
			_bottomWidth->setPercent(std::min(_bottomWidth->getAdjustedValue() + (dialDiff / 2.0f), 1.0f));

			_volume->_computeUniforms["OpacityWidth"]->setElement(_tentIndex, _bottomWidth->getAdjustedValue());
			float width = _tWOnly->_tents.at(_tentIndex)->changeBottomVertices(_bottomWidth->getAdjustedValue());
			_volume->_computeUniforms["OpacityTopWidth"]->setElement(_tentIndex, width);
			_topWidth->setPercent(width);
			bVLabel->setText(std::to_string(_bottomWidth->getAdjustedValue()).substr(0, 4));
		}
	}
	if (ui == _dialTW)
	{
		float dialDiff = _dialTW->getValue();
		if (dialDiff != 0.0) {
			_topWidth->setPercent(std::min(_topWidth->getAdjustedValue() + (dialDiff/2.0f), 1.0f));

			float width = _tWOnly->_tents.at(_tentIndex)->changeTopVertices(_topWidth->getAdjustedValue());
			_volume->_computeUniforms["OpacityTopWidth"]->setElement(_tentIndex, width);
			_topWidth->setPercent(width);
			tVLabel->setText(std::to_string(_topWidth->getAdjustedValue()).substr(0, 4));
		}
	}
	if (ui == _dialBottom)
	{
		float dialDiff = _dialBottom->getValue();
		if (dialDiff != 0.0) {
			_bottom->setPercent(std::min(_bottom->getAdjustedValue() + (dialDiff/2.0f), 1.0f));

			float height = _tWOnly->_tents.at(_tentIndex)->changeBottomHeight(_bottom->getAdjustedValue());
			_volume->_computeUniforms["Lowest"]->setElement(_tentIndex, height);
			_bottom->setPercent(height);
			lVLabel->setText(std::to_string(_bottom->getAdjustedValue()).substr(0, 4));
		}
	}


	_volume->setDirtyAll();
	
}

void TentWindow::initDials() {
	_dialCenter = new Dial(osg::Vec4(0.56, 0.05, 0.25, 1.0));
	_dialCenter->setPercentSize(osg::Vec3(.075, 30, 1));
	_dialCenter->setPercentPos(osg::Vec3(0.087, -6, 0));
	_dialCenter->setCallback(this);

	_dialBW = new Dial(osg::Vec4(0.56, 0.05, 0.25, 1.0));
	_dialBW->setPercentSize(osg::Vec3(.075, 30, 1));
	_dialBW->setPercentPos(osg::Vec3(0.087, -6, 0));
	_dialBW->setCallback(this);

	_dialBottom = new Dial(osg::Vec4(0.56, 0.05, 0.25, 1.0));
	_dialBottom->setPercentSize(osg::Vec3(.075, 30, 1));
	_dialBottom->setPercentPos(osg::Vec3(0.087, -6, 0));
	_dialBottom->setCallback(this);

	_dialHeight = new Dial(osg::Vec4(0.56, 0.05, 0.25, 1.0));
	_dialHeight->setPercentSize(osg::Vec3(.075, 30, 1));
	_dialHeight->setPercentPos(osg::Vec3(0.087, -6, 0));
	_dialHeight->setCallback(this);

	_dialTW = new Dial(osg::Vec4(0.56, 0.05, 0.25, 1.0));
	_dialTW->setPercentSize(osg::Vec3(.075, 30, 1));
	_dialTW->setPercentPos(osg::Vec3(0.087, -6, 0));
	_dialTW->setCallback(this);
}



Tent* TentWindow::addTent(int index, osg::Vec3 color) {
	Tent* tent =new Tent(osg::Vec4(0.1, 0.1, 0.1, 1.0));
	tent->setPercentPos(osg::Vec3(.5, (-index), 0));
	tent->setPercentSize(osg::Vec3(1, 1, 1));
	tent->setColor(color);
	tent->getGeode()->getOrCreateStateSet()->setRenderBinDetails(index, "RenderBin");
	_tWOnly->_bknd->addChild(tent);
	_tWOnly->_tents.push_back(tent);

	_volume->_computeUniforms["OpacityWidth"]->setElement(index, tent->getBottomWidth());
	_volume->_computeUniforms["OpacityTopWidth"]->setElement(index, tent->getTopWidth());
	_volume->_computeUniforms["OpacityCenter"]->setElement(index, tent->getCenter());
	_volume->_computeUniforms["OpacityMult"]->setElement(index, tent->getHeight());
	_volume->_computeUniforms["Lowest"]->setElement(index, tent->getBottom());
	_volume->_computeUniforms["TriangleCount"]->set(float(index+1));
	_volume->setDirtyAll();
	this->setTent(index);
	return tent;
}
void TentWindow::setTent(int index) {
	_tentIndex = index;


	_topWidth->setPercent(_tWOnly->_tents.at(_tentIndex)->getTopWidth());
	tVLabel->setText(std::to_string(_tWOnly->_tents.at(_tentIndex)->getTopWidth()).substr(0, 4));
	_bottomWidth->setPercent(_tWOnly->_tents.at(_tentIndex)->getBottomWidth());
	bVLabel->setText(std::to_string(_tWOnly->_tents.at(_tentIndex)->getBottomWidth()).substr(0, 4));
	_height->setPercent(_tWOnly->_tents.at(_tentIndex)->getHeight());
	hVLabel->setText(std::to_string(_tWOnly->_tents.at(_tentIndex)->getHeight()).substr(0, 4));
	_bottom->setPercent(_tWOnly->_tents.at(_tentIndex)->getBottom());
	lVLabel->setText(std::to_string(_tWOnly->_tents.at(_tentIndex)->getBottom()).substr(0, 4));
	_centerPos->setPercent(_tWOnly->_tents.at(_tentIndex)->getCenter());
	cVLabel->setText(std::to_string(_tWOnly->_tents.at(_tentIndex)->getCenter()).substr(0, 4));
}

void TentWindow::toggleTent(int index) {
	if (_tWOnly->_tents.at(index)->getSavedHeight() == 0.0f) {//If visible
		_tWOnly->_tents.at(index)->setSavedHeight(_tWOnly->_tents.at(index)->getHeight());
		_tWOnly->_tents.at(index)->changeHeight(0.0f);
		_volume->_computeUniforms["OpacityMult"]->setElement(index, 0.0f);
	}
	else {
		_tWOnly->_tents.at(index)->changeHeight(_tWOnly->_tents.at(index)->getSavedHeight());
		_tWOnly->_tents.at(index)->setSavedHeight(0.0f);
		_volume->_computeUniforms["OpacityMult"]->setElement(index, _tWOnly->_tents.at(index)->getHeight());
	}
	_volume->setDirtyAll();
}

void TentWindow::clearTents() {

	while (!_tWOnly->_tents.empty()) {
		_tWOnly->_bknd->getGroup()->removeChild(_tWOnly->_tents.at(0)->getGroup());
		_tWOnly->_tents.erase(_tWOnly->_tents.begin());
	}
	
}

void TentWindow::fillTentDetails(int _triangleIndex, float center, float bottomWidth, float topWidth, float height, float lowest) {
	Tent* tent = _tWOnly->_tents.at(_triangleIndex);
	tent->setPercentPos(osg::Vec3(center, (-_triangleIndex), 0));
	tent->setCenter(center);
	tent->changeBottomVertices(bottomWidth);
	tent->changeTopVertices(topWidth);
	tent->changeHeight(height);
	tent->changeBottomHeight(lowest);

	_volume->_computeUniforms["OpacityWidth"]->setElement(_triangleIndex, tent->getBottomWidth());
	_volume->_computeUniforms["OpacityTopWidth"]->setElement(_triangleIndex, tent->getTopWidth());
	_volume->_computeUniforms["OpacityCenter"]->setElement(_triangleIndex, tent->getCenter());
	_volume->_computeUniforms["OpacityMult"]->setElement(_triangleIndex, tent->getHeight());
	_volume->_computeUniforms["Lowest"]->setElement(_triangleIndex, tent->getBottom());
	_volume->setDirtyAll();
	setTent(_triangleIndex);
}

std::vector<float> TentWindow::getPresetData(int index) {
	std::vector<float> data;
	data.push_back(_tWOnly->_tents.at(index)->getCenter());
	data.push_back(_tWOnly->_tents.at(index)->getBottomWidth()*2);
	data.push_back(_tWOnly->_tents.at(index)->getTopWidth()*2);
	data.push_back(_tWOnly->_tents.at(index)->getHeight());
	data.push_back(_tWOnly->_tents.at(index)->getBottom());
	return data;
}

std::vector<float> TentWindow::getCinePresetData(int index) {
	std::vector<float> data;
	data.push_back(_tWOnly->_tents.at(index)->getHeight());
	data.push_back(_tWOnly->_tents.at(index)->getBottom());
	data.push_back(_tWOnly->_tents.at(index)->getBottomWidth()*2);
	data.push_back(_tWOnly->_tents.at(index)->getTopWidth()*2);
	data.push_back(_tWOnly->_tents.at(index)->getCenter());
	return data;
}

TentWindowOnly::TentWindowOnly() :
	UIElement()
{
	_bknd = new UIQuadElement(osg::Vec4(.04, .25, .4, 0));
	_bknd->setTransparent(false);
	_bknd->setBorderSize(0.01);
	addChild(_bknd);
	_bknd->setPercentSize(osg::Vec3(1, 1, 1.0));

	

	
	_tent = new Tent(osg::Vec4(0.1, 0.1, 0.1, 1.0));
	_tent->getGeode()->getOrCreateStateSet()->setRenderBinDetails(0, "RenderBin");
	_tent->setPercentPos(osg::Vec3(.5, -0.1, 0));
	_tent->setPercentSize(osg::Vec3(1, 0.015, 1));

	

	_bknd->addChild(_tent);

	_tents.push_back(_tent);
	_tentIndex = 0;



	/*UIList* list = new UIList(UIList::TOP_TO_BOTTOM, UIList::CONTINUE);
	list->setPercentPos(osg::Vec3(0, 0, -.25));
	list->setPercentSize(osg::Vec3(1, 1, .75));
	list->setAbsoluteSpacing(1);
	_bknd->addChild(list);*/


}

void TentWindowOnly::setUpGradients() {

}

void TentWindowOnly::uiCallback(UICallbackCaller* ui) {

}

Tent* TentWindowOnly::addTent(int index, osg::Vec3 color) {
	Tent* tent = new Tent(osg::Vec4(0.1, 0.1, 0.1, 1.0));
	tent->setPercentPos(osg::Vec3(.7, (-index), 0));
	tent->setPercentSize(osg::Vec3(1, 0, .25));
	tent->setColor(color);
	tent->getGeode()->getOrCreateStateSet()->setRenderBinDetails(index, "RenderBin");
	_bknd->addChild(tent);
	_tents.push_back(tent);

	_volume->_computeUniforms["OpacityWidth"]->setElement(index, tent->getBottomWidth());
	_volume->_computeUniforms["OpacityTopWidth"]->setElement(index, tent->getTopWidth());
	_volume->_computeUniforms["OpacityCenter"]->setElement(index, tent->getCenter());
	_volume->_computeUniforms["OpacityMult"]->setElement(index, tent->getHeight());
	_volume->_computeUniforms["Lowest"]->setElement(index, tent->getBottom());
	_volume->_computeUniforms["TriangleCount"]->set(float(index+1));
	_volume->setDirtyAll();
	this->setTent(index);
	return tent;
}
void TentWindowOnly::setTent(int index) {
	_tentIndex = index;
	_tents.at(_tentIndex)->setSelected(true);
}

void TentWindowOnly::toggleTent(int index) {
	if (_tents.at(index)->getSavedHeight() == 0.0f) {//If visible
		_tents.at(index)->setSavedHeight(_tents.at(index)->getHeight());
		_tents.at(index)->changeHeight(0.0f);
		_volume->_computeUniforms["OpacityMult"]->setElement(index, 0.0f);
	}
	else {
		_tents.at(index)->changeHeight(_tents.at(index)->getSavedHeight());
		_tents.at(index)->setSavedHeight(0.0f);
		_volume->_computeUniforms["OpacityMult"]->setElement(index, _tents.at(index)->getHeight());
	}
	_volume->setDirtyAll();
}

void TentWindowOnly::clearTents() {
	
	while(!_tents.empty()) {
		_bknd->removeChild(_tents.at(0));
		_tents.erase(_tents.begin());
	}

}

void TentWindowOnly::fillTentDetails(int _triangleIndex, float center, float bottomWidth, float topWidth, float height, float lowest) {
	Tent* tent = _tents.at(_triangleIndex);
	tent->setPercentPos(osg::Vec3(center, (-_triangleIndex), 0));
	tent->setCenter(center);
	tent->changeBottomVertices(bottomWidth);
	tent->changeTopVertices(topWidth);
	tent->changeHeight(height);
	tent->changeBottomHeight(lowest);

	_volume->_computeUniforms["OpacityWidth"]->setElement(_triangleIndex, tent->getBottomWidth());
	_volume->_computeUniforms["OpacityTopWidth"]->setElement(_triangleIndex, tent->getTopWidth());
	_volume->_computeUniforms["OpacityCenter"]->setElement(_triangleIndex, tent->getCenter());
	_volume->_computeUniforms["OpacityMult"]->setElement(_triangleIndex, tent->getHeight());
	_volume->_computeUniforms["Lowest"]->setElement(_triangleIndex, tent->getBottom());
	_volume->setDirtyAll();
	setTent(_triangleIndex);
}

std::vector<float> TentWindowOnly::getPresetData(int index) {
	std::vector<float> data;
	data.push_back(_tents.at(index)->getCenter());
	data.push_back(_tents.at(index)->getBottomWidth());
	data.push_back(_tents.at(index)->getTopWidth());
	data.push_back(_tents.at(index)->getHeight());
	data.push_back(_tents.at(index)->getBottom());
	return data;
}
///////////////Hist
HistQuad::HistQuad() :
	UIElement()
{
	_bknd = new ShaderQuad();
	_bknd->setProgram(getOrLoadProgram());
	addChild(_bknd);

}

osg::Program* HistQuad::getOrLoadProgram() {
	if (!_histprogram)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("hist.frag");
		_histprogram = new osg::Program;
		_histprogram->setName("HistFrag");
		_histprogram->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_histprogram->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _histprogram;
}

void HistQuad::setBB(osg::ref_ptr<osg::ShaderStorageBufferBinding> bb) {
	_bknd->getGeode()->getOrCreateStateSet()->setAttribute(bb, osg::StateAttribute::ON);
}

void HistQuad::setMax(unsigned int histMax) {
 	_bknd->addUniform("histMax", histMax);
}
/////////////////////Histo

float getPt(float n1, float n2, float perc)
{
	float diff = n2 - n1;

	return n1 + (diff * perc);
}


void CurvedQuad::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_geode);
	_transform->addChild(_intersect);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);
	_polyGeom = new osg::Geometry();



	const int res = 100;

	osg::Vec3 curvedCoords[res]; //Top
	osg::Vec3 curvedCoords2[res];	//Bot

	osg::Vec2 containerCoords[] =
	{ 
		osg::Vec2(0.0,0.0),
		osg::Vec2(0.0,0.75),
		osg::Vec2(1.0,0.75),
		osg::Vec2(1.0,0.0)
	};

	//scale
	//osg::Vec2 size(100, 100);
	//for (int i = 0; i < 4; i++) {
	//	containerCoords[i].x() *= size.x();
	//	containerCoords[i].y() *= size.y();
	//}
	


	
	int index = 0;
	for (float i = 0; i < 1; i += 0.01)
	{
		// 1st
		float xa = getPt(containerCoords[0].x(), containerCoords[1].x(), i);
		float ya = getPt(containerCoords[0].y(), containerCoords[1].y(), i);
		float xb = getPt(containerCoords[1].x(), containerCoords[2].x(), i);
		float yb = getPt(containerCoords[1].y(), containerCoords[2].y(), i);
		float xc = getPt(containerCoords[2].x(), containerCoords[3].x(), i);
		float yc = getPt(containerCoords[2].y(), containerCoords[3].y(), i);

		// 2nd
		float xm = getPt(xa, xb, i);
		float ym = getPt(ya, yb, i);
		float xn = getPt(xb, xc, i);
		float yn = getPt(yb, yc, i);

		// 3rd
		float x = getPt(xm, xn, i);
		float y = getPt(ym, yn, i);

		curvedCoords[index] = osg::Vec3(x, 0.0, y);
		index++;
	}

	for (int i = 0; i < res; i++) {
		osg::Vec3 botCoord = curvedCoords[i];
		botCoord.z() -= 1.0;
		botCoord.x() *= .95;
		botCoord.x() += .025;
		curvedCoords2[i] = botCoord;
	}

	osg::ref_ptr<osg::Vec3Array> quadCoords = new osg::Vec3Array();
	quadCoords->push_back(curvedCoords[0]);
	for (int i = 0; i < res-1; i++) {
		osg::Vec3 coord1 = curvedCoords2[i];
		quadCoords->push_back(coord1);
		osg::Vec3 coord2 = curvedCoords[i+1];
		quadCoords->push_back(coord2);
	}

	int numCoords = 0;
	if (_total != 1) { // segment
		osg::ref_ptr<osg::Vec3Array> segmentCoords = new osg::Vec3Array();
		float segmentSize = 1.0 / (float)_total;
		int size = quadCoords->size();

		int index = 0;
		while (index < size  && quadCoords->at(index).x() < segmentSize*(_curr+1) ) {
			if (quadCoords->at(index).x() >= segmentSize* _curr) {
				segmentCoords->push_back(quadCoords->at(index));
			}
			index++;
		}

		//normalize 
		for (int i = 0; i < segmentCoords->size(); i++) {
			segmentCoords->at(i).x() -= segmentSize * _curr ;
			segmentCoords->at(i).x() *= _total ;

		}



		numCoords = segmentCoords->size();
		_polyGeom->setVertexArray(segmentCoords);



	}
	else {
		numCoords = quadCoords->size();
		_polyGeom->setVertexArray(quadCoords);
	}

	
 


	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, numCoords));

	_geode->addDrawable(_polyGeom);
	_intersect->addDrawable(_polyGeom);

	osg::Vec2Array* texcoords = new osg::Vec2Array;
	texcoords->push_back(osg::Vec2(1, 0));
	texcoords->push_back(osg::Vec2(1, 1));
	texcoords->push_back(osg::Vec2(0, 1));
	texcoords->push_back(osg::Vec2(0, 0));

	_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);
	setTransparent(false);

	updateGeometry();


	
	
	
}

void CurvedQuad::updateGeometry() {
	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(_color);
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	((osg::Geometry*)_geode->getDrawable(0))->setVertexAttribArray(2, colors, osg::Array::BIND_OVERALL);

	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);


	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	}
}

void CurvedQuad::setColor(osg::Vec4 color)
{
	
	_colorUniform->set(color);

}

void CurvedQuad::setTransparent(bool transparent)
{
	if (transparent)
	{
 		osg::LineWidth* linewidth = new osg::LineWidth();
		linewidth->setWidth(30.0f);
		_geode->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
		_geode->getOrCreateStateSet()->setRenderBinDetails(5, "RenderBin");
		_geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
		_geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

	}
	else
	{

		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
	}
}


void CurvedQuad::processHover(bool enter)
{
	if (!isOn()) {
		if (enter)
		{
			_colorUniform->set(osg::Vec4(UI_RED_HOVER_COLOR, 1.0));
		}
		else
		{
			_colorUniform->set(UI_BACKGROUND_COLOR);
		}
	}
}


template <typename T>
void CurvedQuad::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void CurvedQuad::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void CurvedQuad::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* CurvedQuad::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void CurvedQuad::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);

}

osg::Program* CurvedQuad::getOrLoadProgram()
{
	/*if (!_triangleProg)
	{*/

	const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
	const std::string frag = HelmsleyVolume::loadShaderFile("curvedQuad.frag");
	_curvedQuadProg = new osg::Program;
	_curvedQuadProg->setName("CurvedQuad");

	_curvedQuadProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_curvedQuadProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));


	//}

	return _curvedQuadProg;
}

//////////////////////Tent
void Tent::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);
	_polyGeom = new osg::Geometry();

	
	osg::Vec3 myCoords[] =
	{
		osg::Vec3(rightPointX, 0.0, -1.0),
		osg::Vec3(rightPointX, 0.0, bottomHeight),
		osg::Vec3(topPointX, 0.0, height),
		osg::Vec3(-topPointX, 0.0, height),
		osg::Vec3(leftPointX, 0.0, bottomHeight),
		osg::Vec3(leftPointX, 0.0, -1.0)

	};
	int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);

	osg::Vec3Array* vertices = new osg::Vec3Array(numCoords, myCoords);

	
	_polyGeom->setVertexArray(vertices);

	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, numCoords));

	_geode->addDrawable(_polyGeom);


	osg::Vec2Array* texcoords = new osg::Vec2Array;
	texcoords->push_back(osg::Vec2(1, 0));
	texcoords->push_back(osg::Vec2(1, bottomHeight));
	texcoords->push_back(osg::Vec2(0.5 + (topPointX/2.0), 1));
	texcoords->push_back(osg::Vec2(0.5 - (topPointX/2.0), 1));
	texcoords->push_back(osg::Vec2(0, bottomHeight));
	texcoords->push_back(osg::Vec2(0, 0));

	


	_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);
	setTransparent(false);

	updateGeometry();
}

void Tent::updateGeometry()
{

	osg::Vec3 myCoords[] =
	{
		osg::Vec3(rightPointX, 0.0, -1.0),
		osg::Vec3(rightPointX, 0.0, bottomHeight),
		osg::Vec3(topPointX, 0.0, height),
		osg::Vec3(-topPointX, 0.0, height),
		osg::Vec3(leftPointX, 0.0, bottomHeight),
		osg::Vec3(leftPointX, 0.0, -1.0)

	};
	int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);

	osg::Vec3Array* vertices = new osg::Vec3Array(numCoords, myCoords);
	_polyGeom->setVertexArray(vertices);

	osg::Vec2Array* texcoords = new osg::Vec2Array;	
	texcoords->push_back(osg::Vec2(1, 0));
	texcoords->push_back(osg::Vec2(1, bottomHeight));
	texcoords->push_back(osg::Vec2(.5 + ((topPointX / rightPointX) / 2.0), 1));
	texcoords->push_back(osg::Vec2(.5 - ((topPointX / rightPointX) / 2.0), 1));
	texcoords->push_back(osg::Vec2(0, bottomHeight));
	texcoords->push_back(osg::Vec2(0, 0));



	_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);

	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(_color);
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	((osg::Geometry*)_geode->getDrawable(0))->setVertexAttribArray(2, colors, osg::Array::BIND_OVERALL);

	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);


	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	}
}

void Tent::setColor(osg::Vec3 color)
{
	
	_colorUniform->set(color);
	
}

void Tent::setTransparent(bool transparent)
{
	if (transparent)
	{
		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
		
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
		_geode->getOrCreateStateSet()->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA));
		
	}
	else
	{

		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
	}
}

void Tent::setRounding(float absRounding, float percentRounding)
{
	_absoluteRounding->set(absRounding);
	_percentRounding->set(percentRounding);
}


template <typename T>
void Tent::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Tent::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Tent::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* Tent::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void Tent::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
	
}

osg::Program* Tent::getOrLoadProgram()
{
	if (!_triangleProg)
	{
		
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("triangle.frag");
		_triangleProg = new osg::Program;
		_triangleProg->setName("Triangle");
		
		_triangleProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_triangleProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));

		
	}

	return _triangleProg;
}

float Tent::changeBottomVertices(float x) {
	rightPointX = x;
	leftPointX = -x;
	topPointX = std::min(actualTop, rightPointX);
	_widthUniform->set(x*2);
	updateGeometry();
	return topPointX;
}

float Tent::changeTopVertices(float x) {
	actualTop = x;
	topPointX = std::min(x, rightPointX);
	updateGeometry();
	return topPointX;
}

float Tent::changeHeight(float x) {
	height = x - 1.0;
	bottomHeight = std::min(actualBottomHeight, height);
	updateGeometry();
	return bottomHeight + 1.0;
}

float Tent::changeBottomHeight(float x) {
	actualBottomHeight = x - 1.0;
	if (actualBottomHeight >= height)
	{
		bottomHeight = height - .001;
	}
	else {
		bottomHeight = actualBottomHeight;
	}
	updateGeometry();
	return bottomHeight + 1.0;
}
//////////////////////////////////////////////////////////////////////////////////centerline

void Line::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);
	_polyGeom = new osg::Geometry();



	int numCoords = _coords->size();

	
	_polyGeom->setVertexArray(_coords);

	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, numCoords));
	

	_geode->addDrawable(_polyGeom);

	setTransparent(false);

	updateGeometry();
}

void Line::updateGeometry()
{


	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(_color);
	/*for (int i = 0; i < _coords->size(); i++) {
		colors->push_back(_color);
	}*/

	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	((osg::Geometry*)_geode->getDrawable(0))->setVertexAttribArray(2, colors, osg::Array::BIND_OVERALL);
	

	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);


	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);
	

	}
}

void Line::setColor(osg::Vec3 color)
{

	_colorUniform->set(color);

}

void Line::setTransparent(bool transparent)
{
	if (transparent)
	{
		

	

		_geode->getOrCreateStateSet()->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
		osg::LineWidth* linewidth = new osg::LineWidth();
		linewidth->setWidth(30.0f);
		_geode->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON);
		_geode->getOrCreateStateSet()->setRenderBinDetails(5, "RenderBin");
		_geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF|osg::StateAttribute::OVERRIDE);
		_geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
	}
	else
	{

		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
	}
}




template <typename T>
void Line::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Line::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Line::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* Line::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void Line::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
}

osg::Program* Line::getOrLoadProgram()
{
	/*if (!_triangleProg)
	{*/

	const std::string vert = HelmsleyVolume::loadShaderFile("lines.vert");
	const std::string frag = HelmsleyVolume::loadShaderFile("lines.frag");
	_lineProg = new osg::Program;
	_lineProg->setName("Lines");

	_lineProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_lineProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));


	//}

	return _lineProg;
}


void MarchingCubesRender::createGeometry()
{
	_polyGeom = new osg::Geometry();

	int numFloats = _coords->size();
	osg::ref_ptr<osg::Vec3Array> vec3Coords = new osg::Vec3Array();
	osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;


	for (int i = 0; i < numFloats; i ++) {
		if (_coords->at(i).x() == std::numeric_limits<float>::max()) {
			std::cout << "index: " << i << std::endl;
			break;
		}
		//std::cout << "coords: " << _coords->at(i) << std::endl;
		osg::Vec3 vec3;
		vec3.x() = _coords->at(i).x();
		//i++;
		vec3.y() = _coords->at(i).y();
		//i++;
		vec3.z() = _coords->at(i).z();

 		/*vec3.x() = (vec3.x() / _voldims.x()) - .5f;
		vec3.y() = ((vec3.y() / _voldims.y())-.5f);
		vec3.z() = (vec3.z() / _voldims.z())-.5f;*/

		vec3Coords->push_back(vec3);
		
		int currsize = vec3Coords->size();
				//NORMALS//
		if (currsize % 3 == 0) {
			osg::Vec3 p1 = vec3Coords->at(currsize - 3); p1.x() = (p1.x()+.5)*_voldims.x(); p1.y() = (p1.y()+.5)*_voldims.y(); p1.z() = (p1.z()+.5)*_voldims.z();
			osg::Vec3 p2 = vec3Coords->at(currsize - 2); p2.x() = (p2.x() + .5) * _voldims.x(); p2.y() = (p2.y() + .5) * _voldims.y(); p2.z() = (p2.z() + .5) * _voldims.z();
			osg::Vec3 p3 = vec3Coords->at(currsize - 1); p3.x() = (p3.x() + .5) * _voldims.x(); p3.y() = (p3.y() + .5) * _voldims.y(); p3.z() = (p3.z() + .5) * _voldims.z();

			osg::Vec3 vecA = p2 - p1;
			osg::Vec3 vecB = p3 - p2;
			
			osg::Vec3 normal = vecA ^ vecB;
			normal.normalize();
			normal.x() = (normal.x() / 2.0f) + .5f;
			normal.y() = (normal.y() / 2.0f) + .5f;
			normal.z() = (normal.z() / 2.0f) + .5f;

			colors->push_back(osg::Vec4(normal, 1.0));
			colors->push_back(osg::Vec4(normal, 1.0));
			colors->push_back(osg::Vec4(normal, 1.0));
		}
	}

	int numCoords = vec3Coords->size();
 	
  	
	


 	_polyGeom->setVertexArray(vec3Coords);
	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numCoords));
	_geode->addDrawable(_polyGeom);
	
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
	((osg::Geometry*)_geode->getDrawable(0))->setVertexAttribArray(2, colors, osg::Array::BIND_PER_VERTEX);
	

	updateGeometry();
}

void MarchingCubesRender::updateGeometry()
{

	_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
	_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
	_geode->getOrCreateStateSet()->setAttributeAndModes(_ssbb, osg::StateAttribute::ON);


	_geode->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
	_geode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);


 	if (_program.valid())
	{
 		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);
		std::cout << "program applied" << std::endl;
	}
	 
}


template <typename T>
void MarchingCubesRender::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void MarchingCubesRender::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void MarchingCubesRender::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* MarchingCubesRender::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void MarchingCubesRender::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
}

osg::Program* MarchingCubesRender::getOrLoadProgram()
{
	if (!_mcProg)
	{

	const std::string vert = HelmsleyVolume::loadShaderFile("mc.vert");
	const std::string frag = HelmsleyVolume::loadShaderFile("mc.frag");
	_mcProg = new osg::Program;
	_mcProg->setName("MCRender");

	_mcProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_mcProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	

	}

	return _mcProg;
}


void TriangleButton::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_intersect);
	_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);
	_polyGeom = new osg::Geometry();


	osg::Vec3 myCoords[] =
	{
		osg::Vec3(-0.5, 0.0, -0.5),
		osg::Vec3(0.5, 0.0, 0.0),
		osg::Vec3(-0.5, 0.0, 0.5)
	};
	_rot = osg::Matrix::rotate(0, 0, 1, 0);
	
	
	
	int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);

	for (int i = 0; i < numCoords; i++) {
		myCoords[i] = _rot * myCoords[i];
	}

	osg::Vec3Array* vertices = new osg::Vec3Array(numCoords, myCoords);


	_polyGeom->setVertexArray(vertices);

	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES, 0, numCoords));

	_geode->addDrawable(_polyGeom);
	_intersect->addDrawable(_polyGeom);
	osg::Vec2Array* texcoords = new osg::Vec2Array;
	texcoords->push_back(osg::Vec2(0, 0));
	texcoords->push_back(osg::Vec2(0, 1));
	texcoords->push_back(osg::Vec2(1, 1));
	texcoords->push_back(osg::Vec2(1, 0));




	_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);

	updateGeometry();
}

void TriangleButton::updateGeometry()
{

	osg::Vec3 myCoords[] =
	{
		osg::Vec3(-0.5, 0.0, -0.5),
		osg::Vec3(0.5, 0.0, 0.0),
		osg::Vec3(-0.5, 0.0, 0.5)
	};


	
	



	int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);


	for (int i = 0; i < numCoords; i++) {
		myCoords[i] = _rot * myCoords[i];
	}


	osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array(numCoords, myCoords);
	_polyGeom->setVertexArray(vertices);

	/*osg::Vec2Array* texcoords = new osg::Vec2Array;
	texcoords->push_back(osg::Vec2(0, 0));
	texcoords->push_back(osg::Vec2(1, 0));
	texcoords->push_back(osg::Vec2(0, 1));*/

	//_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);

	osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
	colors->push_back(_color);
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	((osg::Geometry*)_geode->getDrawable(0))->setVertexAttribArray(2, colors, osg::Array::BIND_OVERALL);

	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);
	


	if (_program.valid())
	{
		_geode->getDrawable(0)->getOrCreateStateSet()->setAttributeAndModes(_program.get(), osg::StateAttribute::ON);

	}
}

void TriangleButton::setRotate(double radians) {
	_rot = osg::Matrix::rotate(osg::PI*radians, 0, 1, 0);
	updateGeometry();
}

void TriangleButton::setColor(osg::Vec3 color)
{

	_colorUniform->set(color);

}



template <typename T>
void TriangleButton::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void TriangleButton::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void TriangleButton::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* TriangleButton::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void TriangleButton::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);

}

osg::Program* TriangleButton::getOrLoadProgram()
{
	/*if (!_triangleProg)
	{*/

	const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
	const std::string frag = HelmsleyVolume::loadShaderFile("triangleButton.frag");
	_triangleButtonProg = new osg::Program;
	_triangleButtonProg->setName("TriangleButton");

	_triangleButtonProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
	_triangleButtonProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));


	//}

	return _triangleButtonProg;
}

bool TriangleButton::processEvent(cvr::InteractionEvent* event)
{
	TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();
	if (tie && tie->getButton() == 0)
	{
		if (tie->getInteraction() == BUTTON_DOWN)
		{
			
			if (_callback)
			{
				_callback->uiCallback(this);
			}
			

		}
		/*else if (tie->getInteraction() == BUTTON_UP && holdingDial)
		{
			holdingDial = false;
			_jump = false;
		}*/
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////////////
void Dial::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_intersect);
	_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);
	_polyGeom = new osg::Geometry();

	
	
	osg::Vec3 myCoords[] =
	{
		osg::Vec3(rightPointX, 0.0, -1.0),
		osg::Vec3(topPointX, 0.0, height),
		osg::Vec3(topPointX, 0.0, height),
		osg::Vec3(leftPointX, 0.0, -1.0)

	};
	int numCoords = sizeof(myCoords) / sizeof(osg::Vec3);

	osg::Vec3Array* vertices = new osg::Vec3Array(numCoords, myCoords);


	_polyGeom->setVertexArray(vertices);

	_polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, numCoords));


	_sphere = new osg::ShapeDrawable(new osg::Sphere());
	_geode->addDrawable(_sphere);
	_intersect->addDrawable(_sphere);

	osg::Vec2Array* texcoords = new osg::Vec2Array;
	texcoords->push_back(osg::Vec2(1, 0));
	texcoords->push_back(osg::Vec2(0.5 + (topPointX / 2.0), 1));
	texcoords->push_back(osg::Vec2(0.5 - (topPointX / 2.0), 1));
	texcoords->push_back(osg::Vec2(0, 0));



	_polyGeom->setVertexAttribArray(1, texcoords, osg::Array::BIND_PER_VERTEX);
	setTransparent(true);

	updateGeometry();
}

void Dial::updateGeometry()
{

	

	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(_color);
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	
	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);


}




bool Dial::processEvent(cvr::InteractionEvent* event)
{
	TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();
	if (tie && tie->getButton() == 0)
	{
		if (tie->getInteraction() == BUTTON_DOWN && !holdingDial)
		{
			holdingDial = true;
			if (_callback)
			{
				_callback->uiCallback(this);
			}
			_startMat = TrackingManager::instance()->getHandMat(tie->getHand());
		
		
		}
		if (tie->getInteraction() == BUTTON_DRAG && holdingDial)
		{
			osg::Matrix currMat = TrackingManager::instance()->getHandMat(tie->getHand());
			
			osg::Quat startQuat = _startMat.getRotate();
			osg::Quat currQuat = currMat.getRotate();
			osg::Quat diff = currQuat * startQuat.osg::Quat::inverse();
			float sYN = startQuat.y() + 1.0f;
			float cYN = currQuat.y() + 1.0f;
			float diffF = cYN - sYN;


			if (_callback)
			{
				if (diffF < .0025 && diffF> -.0025) {
					diffF = 0;
				}
			
				if (diffF > 1.00 || diffF < -1.00) {
					_jump = _jump ? false : true;
					diffF = _value;
				}
				if (_jump) {
					diffF = -diffF;
				}
					
				_value = diffF;
				_callback->uiCallback(this);

			}
			_startMat = currMat;
			
		}
		else if (tie->getInteraction() == BUTTON_UP && holdingDial)
		{
			holdingDial = false;
			_jump = false;
		}
	}
	return holdingDial;
}

float Dial::getValue() {
	return _value;
}
void Dial::setColor(osg::Vec4 color)
{
	if (_color != color)
	{
		_color = color;
		_dirty = true;
	}
}

void Dial::setTransparent(bool transparent)
{
	if (transparent)
	{
		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
	}
	else
	{
		_geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::OPAQUE_BIN);
		_geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::OFF);
	}
}

void Dial::setRounding(float absRounding, float percentRounding)
{
	_absoluteRounding->set(absRounding);
	_percentRounding->set(percentRounding);
}


template <typename T>
void Dial::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Dial::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void Dial::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* Dial::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void Dial::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
}

osg::Program* Dial::getOrLoadProgram()
{
	if (!_dialProg)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("triangle.frag");
		_dialProg = new osg::Program;
		_dialProg->setName("Triangle");
		_dialProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_dialProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _dialProg;
}

void Dial::changeBottomVertices(float x) {
	rightPointX = x;
	leftPointX = -x;
	topPointX = std::min(actualTop, rightPointX);
	updateGeometry();
}

float Dial::changeTopVertices(float x) {
	actualTop = x;
	topPointX = std::min(x, rightPointX);
	updateGeometry();
	return topPointX;
}

void Dial::changeHeight(float x) {
	height = x - 1.0;
	updateGeometry();
}


void ColorSlider::createGeometry()
{
	_transform = new osg::MatrixTransform();
	_intersect = new osg::Geode();

	_group->addChild(_transform);
	_transform->addChild(_intersect);
	_transform->addChild(_geode);

	_intersect->setNodeMask(cvr::INTERSECT_MASK);

	_sphere = new osg::ShapeDrawable(new osg::Sphere());
	
	_geode->addDrawable(_sphere);
	_intersect->addDrawable(_sphere);


	updateGeometry();
}

void ColorSlider::updateGeometry()
{

	

	osg::Vec4Array* colors = new osg::Vec4Array;
	colors->push_back(_color);
	((osg::Geometry*)_geode->getDrawable(0))->setColorArray(colors, osg::Array::BIND_OVERALL);
	
	osg::Matrix mat = osg::Matrix();
	mat.makeScale(_actualSize);
	mat.postMultTranslate(_actualPos);
	_transform->setMatrix(mat);


}


void ColorSlider::uiCallback(UICallbackCaller* ui) {
}

bool ColorSlider::processEvent(cvr::InteractionEvent* event)
{
	TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();
	if (tie && tie->getButton() == 0)
	{
		if (tie->getInteraction() == BUTTON_DOWN)
		{
			if (_callback)
			{
				_callback->uiCallback(this);
			}

		
		
		}
	}
	return false;
}


void ColorSlider::setColor(osg::Vec4 color)
{
	if (_color != color)
	{
		_color = color;
		_dirty = true;
		updateGeometry();
	}
}



template <typename T>
void ColorSlider::addUniform(std::string uniform, T initialvalue)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), initialvalue);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void ColorSlider::addUniform(std::string uniform)
{
	_uniforms[uniform] = new osg::Uniform(uniform.c_str(), 0.0f);
	_geode->getOrCreateStateSet()->addUniform(_uniforms[uniform]);
}

void ColorSlider::addUniform(osg::Uniform* uniform)
{
	_uniforms[uniform->getName()] = uniform;
	_geode->getOrCreateStateSet()->addUniform(uniform);
}

osg::Uniform* ColorSlider::getUniform(std::string uniform)
{
	return _uniforms[uniform];
}

void ColorSlider::setShaderDefine(std::string name, std::string define, osg::StateAttribute::Values on)
{
	_geode->getOrCreateStateSet()->setDefine(name, define, on);
}

osg::Program* ColorSlider::getOrLoadProgram()
{
	if (!_colorSlideProg)
	{
		const std::string vert = HelmsleyVolume::loadShaderFile("transferFunction.vert");
		const std::string frag = HelmsleyVolume::loadShaderFile("triangle.frag");
		_colorSlideProg = new osg::Program;
		_colorSlideProg->setName("Triangle");
		_colorSlideProg->addShader(new osg::Shader(osg::Shader::VERTEX, vert));
		_colorSlideProg->addShader(new osg::Shader(osg::Shader::FRAGMENT, frag));
	}

	return _colorSlideProg;
}

#pragma region ColorPicker

ColorPicker::ColorPicker() :
	UIElement()
{
	color = osg::Vec3(0, 1, 1);

	_bknd = new UIQuadElement(UI_BACKGROUND_COLOR);
	addChild(_bknd);
	
	float border = 20;

	_hue = new ColorPickerHue();
	_hue->setPos(osg::Vec3(0.8f, 0.0f, 0.0f), osg::Vec3(border, -0.1f, -border));
	_hue->setSize(osg::Vec3(0.2f, 1.0f, 1.0f), osg::Vec3(-border*2, 0, -border * 2));
	_bknd->addChild(_hue);
	_hue->setCallback(this);

	_sv = new ColorPickerSaturationValue();
	_sv->setAbsolutePos(osg::Vec3(border, -0.1f, -border));
	_sv->setSize(osg::Vec3(0.8f, 1.0f, 1.0f), osg::Vec3(-border * 2, 0, -border * 2));
	_bknd->addChild(_sv);
	_sv->setCallback(this);
	

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

	//osg::Vec3 solidCol = ColorPicker::returnColor();
	
	//_transferFunction = "vec3(" + std::to_string(solidCol.x()) + "," + std::to_string(solidCol.y()) + "," + std::to_string(solidCol.z()) + ");";

	
}

void ColorPicker::setButton(cvr::UIQuadElement* target) {
	_target = target;

}

//From https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72
osg::Vec3 ColorPicker::RGBtoHSV(float r, float g, float b) {
	float fH;
	float fS;
	float fV;
	
	float fR = r;
	float fG = g;
	float fB = b;


	float fCMax = std::max(std::max(fR, fG), fB);
	float fCMin = std::min(std::min(fR, fG), fB);
	float fDelta = fCMax - fCMin;

	if (fDelta > 0) {
		if (fCMax == fR) {
			fH = 60 * (fmod(((fG - fB) / fDelta), 6));
		}
		else if (fCMax == fG) {
			fH = 60 * (((fB - fR) / fDelta) + 2);
		}
		else if (fCMax == fB) {
			fH = 60 * (((fR - fG) / fDelta) + 4);
		}

		if (fCMax > 0) {
			fS = fDelta / fCMax;
		}
		else {
			fS = 0;
		}

		fV = fCMax;
	}
	else {
		fH = 0;
		fS = 0;
		fV = fCMax;
	}

	if (fH < 0) {
		fH = 360 + fH;
	}
	osg::Vec3 hsv = osg::Vec3(fH/360, fS, fV);
	return hsv;
}

osg::Vec3 ColorPicker::returnColor() {
	//hsv to rgb
	double H = color.x() * 360.0;
	double S = color.y();
	double V = color.z();

	double C = S * V;
	double X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	double m = V - C;
	double Rs, Gs, Bs;

	if (H >= 0 && H < 60 ) {
		Rs = C;
		Gs = X;
		Bs = 0;
	}
	else if (H >= 60  && H < 120 ) {
		Rs = X;
		Gs = C;
		Bs = 0;
	}
	else if (H >= 120  && H < 180 ) {
		Rs = 0;
		Gs = C;
		Bs = X;
	}
	else if (H >= 180 && H < 240 ) {
		Rs = 0;
		Gs = X;
		Bs = C;
	}
	else if (H >= 240  && H < 300 ) {
		Rs = X;
		Gs = 0;
		Bs = C;
	}
	else {
		Rs = C;
		Gs = 0;
		Bs = X;
	}

	rgbColor.x() = (Rs + m);
	rgbColor.y() = (Gs + m);
	rgbColor.z() = (Bs + m);
	return rgbColor;
}

Selection::Selection(std::string name, bool hasMask) :
	cvr::UIElement()
{
	using namespace cvr;
	_uiTexture = nullptr;
	_name = name;
	std::string copy = name;
	copy.erase(copy.begin());
	_bknd = new cvr::UIQuadElement(UI_WHITE_COLOR);
	addChild(_bknd);
	_bknd->setPercentPos(osg::Vec3(0, -1, 0));
	_bknd->setBorderSize(0.01);
	_uiText = new UIText(copy, 35.0f, osgText::TextBase::LEFT_CENTER);
	_uiText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_uiText->setPercentPos(osg::Vec3(0.0, -1.0, -1.0));
	_uiText->setPercentSize(osg::Vec3(1.0, 0.0, 0.08));

	if (hasMask)
		addMaskSymbol();


	_bknd->addChild(_uiText);

	_button = new CallbackButton();
	_button->setCallback(this);
	_button->setId(name);
	_bknd->addChild(_button);
	
}

void Selection::addMaskSymbol() {
	if (_uiText->getChild(0) == NULL) {
		cvr::UIQuadElement* symbol = new cvr::UIQuadElement(UI_BLACK_COLOR);
		_uiText->addChild(symbol);
		symbol->setPercentPos(osg::Vec3(0.87, 1.0, -0.05));
		symbol->setPercentSize(osg::Vec3(0.08, 1.0, 1.0));
	}
}
void Selection::removeMaskSymbol() {
	if (!_uiText->getChild(0) == NULL) {
		_uiText->removeChild(_uiText->getChild(0));
	}
}
void Selection::setMask(bool hasMask) {
	if (hasMask) {
		addMaskSymbol();
	}
	else {
		removeMaskSymbol();
	}
}


bool Selection::processEvent(cvr::InteractionEvent* event) {
	return false;
}

void Selection::uiCallback(UICallbackCaller* ui) {
	
}

void Selection::setName(std::string name) {
	_name = name;
	std::string copy = name;
	copy.erase(copy.begin());
	_uiText->setText(copy);
	osgText::Text* text = _uiText->getTextObject();
	if (_name.size() > 20){
		text->setCharacterSize(30.0f);
	}
	else {
		text->setCharacterSize(40.0f);
	}
	
}

void Selection::lowerTextSize() {
	osgText::Text* text = _uiText->getTextObject();
	text->setCharacterSize(30.0f);
}

void Selection::setImage(UITexture* uiTexture) {

	_uiTexture = uiTexture;
	_bknd->addChild(_uiTexture);
}

void Selection::removeImage() {
	if(_uiTexture!=nullptr)
		_bknd->removeChild(_uiTexture);
}
#pragma endregion

#pragma region FullButton
FullButton::FullButton(std::string txt, osg::Vec4 color, osg::Vec4 color2) :
	cvr::UIElement(), _originalColor(color), _savedColor(color2)
{
	_bknd = new cvr::UIQuadElement(color);
	
	

	addChild(_bknd);
	_bknd->setPercentPos(osg::Vec3(0, -1, 0));

	_uiText = new UIText(txt, 50.f, osgText::TextBase::CENTER_CENTER);
	_uiText->setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
	_bknd->addChild(_uiText);

	_button = new HoverButton(_bknd, color, color2);
	_button->setCallback(this);
 	_bknd->addChild(_button);

	if (color2.x() == -1) {
		_savedColor == _originalColor;
	}
}


bool FullButton::processEvent(cvr::InteractionEvent* event) {
	return false;
}

void FullButton::uiCallback(UICallbackCaller* ui) {

}


void FullButton::processHover(bool enter)
{
	
	if (enter)
	{
		
		_bknd->setColor(_savedColor);
	}
	else
	{
		_bknd->setColor(_originalColor);
		
	}
	std::cout << "on hover" << std::endl;
}

#pragma endregion