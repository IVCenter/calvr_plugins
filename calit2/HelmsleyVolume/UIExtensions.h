#ifndef UI_EXTENSIONS_H
#define UI_EXTENSIONS_H

#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UICheckbox.h>
#include <cvrMenu/NewUI/UIToggle.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/NewUI/UIText.h>
#include <cvrMenu/NewUI/UIRadial.h>
#include <cvrMenu/NewUI/UIList.h>
#include <cvrMenu/NewUI/UIQuadElement.h>

#include <cvrConfig/ConfigManager.h>

class UICallback;
class UICallbackCaller;

//Callback class (NewUI doesnt implement callback by default - extended classes will for now just to make things easier
class UICallback
{
public:
	UICallback() {}

	virtual void uiCallback(UICallbackCaller* ui) = 0;
};

class UICallbackCaller
{
public:
	UICallbackCaller() 
	{
		_callback = NULL;
	}

	void setCallback(UICallback* callback) { _callback = callback; }

protected:
	UICallback* _callback;

};

class VisibilityToggle : public cvr::UIToggle, public UICallbackCaller
{
public:

	VisibilityToggle(std::string text);

	virtual bool onToggle() override;

	cvr::UICheckbox* eye;
	cvr::UIText* label;
};

class CallbackRadial : public cvr::UIRadial, public UICallbackCaller
{
public:

	virtual void onSelectionChange() override;
};

class ToolSelector : public cvr::UIList
{
public:
	ToolSelector(Direction d = LEFT_TO_RIGHT, OverflowBehavior o = CUT);

protected:
	CallbackRadial* _toolRadial;
};

class CallbackSlider : public cvr::UISlider, public UICallbackCaller
{
public:
	bool onPercentChange() override;

	void setMax(float max);
	void setMin(float min);
	float getMax();
	float getMin();
	float getAdjustedValue() { return _min + (_max - _min) * _percent; }

protected:
	float _max = 1;
	float _min = 0;
};

class ShaderQuad : public cvr::UIQuadElement
{
public:
	ShaderQuad()
		: UIQuadElement(osg::Vec4(1, 1, 1, 1))
	{
		_uniforms = std::map<std::string, osg::Uniform*>();
	}

	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }
	virtual void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);

	virtual void updateGeometry() override;

protected:
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;
};

#endif