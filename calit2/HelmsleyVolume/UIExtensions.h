#ifndef UI_EXTENSIONS_H
#define UI_EXTENSIONS_H

#define UI_BACKGROUND_COLOR osg::Vec4(0.5, 0.5, 0.5, 1)
#define UI_ACTIVE_COLOR osg::Vec4(0.8, 1, 0.8, 1)
#define UI_INACTIVE_COLOR osg::Vec4(1, 0.8, 0.8, 1)
#define UI_BLUE_COLOR osg::Vec4(0.8, 0.9, 1.0, 1)

#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UICheckbox.h>
#include <cvrMenu/NewUI/UIToggle.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/NewUI/UIText.h>
#include <cvrMenu/NewUI/UIRadial.h>
#include <cvrMenu/NewUI/UIList.h>
#include <cvrMenu/NewUI/UIQuadElement.h>

#include <cvrConfig/ConfigManager.h>

#include "VolumeGroup.h"//GA

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

class CallbackButton
	: public cvr::UIButton, public UICallbackCaller
{
public:
	virtual bool onButtonPress(bool pressed)
	{
		if (pressed)
		{
			if (_callback)
			{
				_callback->uiCallback(this);
			}
			return true;
		}
		return false;
	}
};

class CallbackToggle : public cvr::UIToggle, public UICallbackCaller
{
public:

	virtual bool onToggle() override;

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

class ToolRadialButton : public cvr::UIRadialButton
{
public:
	ToolRadialButton(cvr::UIRadial* parent, std::string iconpath);

	cvr::UITexture* getIcon() { return _icon; }
	void setIcon(std::string iconpath) { _icon->setTexture(iconpath); }

	virtual void processHover(bool enter) override;

protected:
	cvr::UITexture* _icon;
	cvr::UIQuadElement* _quad;
};

class ToolToggle : public CallbackToggle
{
public:
	ToolToggle(std::string iconpath);

	cvr::UITexture* getIcon() { return _icon; }
	void setIcon(std::string iconpath) { _icon->setTexture(iconpath); }

	virtual void processHover(bool enter) override;

protected:
	cvr::UITexture* _icon;
	cvr::UIQuadElement* _quad;
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

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);

	virtual void updateGeometry() override;

protected:
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;
};


class PlanePointer : public cvr::UIElement
{
public:
	virtual void createGeometry();
	virtual void updateGeometry();

	virtual bool processEvent(cvr::InteractionEvent* event);
	virtual bool onPosChange() { return true; }

	virtual void setPointer(float x, float y);
	virtual osg::Vec2 getPointer() { return _pointer; }

protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;

	osg::Vec2 _pointer;

	unsigned int _button;
	bool _held;
};


class ColorPickerSaturationValue : public PlanePointer, public UICallbackCaller
{
public:
	ColorPickerSaturationValue();

	osg::Vec2 getSV() { return _sv; }
	void setHue(float hue);
	void setSV(osg::Vec2 sv) { _sv = sv; }
	void set_indicator();
	virtual bool onPosChange() override;

protected:
	static osg::Program* getOrLoadProgram();
	static osg::Program* _svprogram;

	UIElement* _indicator;
	ShaderQuad* _shader;

private:
	osg::Vec2 _sv;
	float _hue;
};

class ColorPickerHue : public PlanePointer, public UICallbackCaller
{
public:
	ColorPickerHue();

	float getHue() { return _hue; }
	void setSV(osg::Vec2 SV);
	void setHue(float hue) { _hue = hue; }
	void set_indicator();
	virtual bool onPosChange() override;


protected:
	static osg::Program* getOrLoadProgram();
	static osg::Program* _hueprogram;

	UIElement* _indicator;
	ShaderQuad* _shader;

private:
	osg::Vec2 _sv;
	float _hue;
};


class Tent : public cvr::UIElement
{
public:
	Tent(osg::Vec4 color = osg::Vec4(1, 1, 1, 1))
		: UIElement()
	{
		leftPointX = -.25;
		rightPointX = .25;
		topPointX = 0.0;
		actualTop = 0.0;
		height = 0.0;
		bottomHeight = -1.0;
		actualBottomHeight = -1.0;
		_center = .7;
		_color = color;
		_geode = new osg::Geode();


		
		// add the stateset tor the drawable
		

		createGeometry();
	
		_absoluteRounding = new osg::Uniform("absoluteRounding", 0.0f);
		_percentRounding = new osg::Uniform("percentRounding", 0.0f);
		_centerUniform = new osg::Uniform("Center", 0.7f);
		_widthUniform = new osg::Uniform("Width", 0.5f);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_absoluteRounding);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_percentRounding);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_centerUniform);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_widthUniform);
		this->setTransparent(true);
		
		setProgram(getOrLoadProgram());
		addUniform("SV", osg::Vec2(1.0f, 1.0f));
	}

	float changeBottomVertices(float x);
	float changeTopVertices(float x);
	float changeHeight(float x);
	float changeBottomHeight(float x);
	void setCenter(float x) { _center = x; _centerUniform->set(x);}
	float getBottomWidth() {return rightPointX;}
	float getTopWidth() {return topPointX;}
	float getCenter() {return _center;}
	float getHeight() {return height + 1.0;}
	float getBottom() {return bottomHeight + 1.0;}
	
	virtual void createGeometry();
	virtual void updateGeometry();




	virtual void setColor(osg::Vec4 color);

	virtual void setTransparent(bool transparent);

	virtual void setRounding(float absRounding, float percentRounding);



	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);




protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;

	static osg::Program* getOrLoadProgram();
	static osg::Program* _triangleProg;

	osg::Vec4 _color;
	osg::Uniform* _absoluteRounding;
	osg::Uniform* _percentRounding;
	osg::Uniform* _centerUniform;
	osg::Uniform* _widthUniform;

	float leftPointX;
	float rightPointX;
	float topPointX;
	float height;
	float actualTop; 
	float actualBottomHeight; 
	float bottomHeight;
	float _center;
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};

class Dial : public cvr::UIElement, public UICallbackCaller
{
public:
	Dial(osg::Vec4 color = osg::Vec4(1, 1, 1, 1))
		: UIElement()
	{
		holdingDial = false;
		leftPointX = -.25;
		rightPointX = .25;
		topPointX = 0.0;
		actualTop = 0.0;
		height = 0.0;
		_color = color;
		_geode = new osg::Geode();
		createGeometry();
		_jump = false;
		/*_absoluteRounding = new osg::Uniform("absoluteRounding", 0.0f);
		_percentRounding = new osg::Uniform("percentRounding", 0.0f);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_absoluteRounding);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_percentRounding);*/


		setProgram(getOrLoadProgram());
		addUniform("SV", osg::Vec2(1.0f, 1.0f));
	}
	float getValue();
	void changeBottomVertices(float x);
	float changeTopVertices(float x);
	void changeHeight(float x);
	virtual void createGeometry();
	virtual void updateGeometry();

	virtual bool processEvent(cvr::InteractionEvent* event) override;

	virtual void setColor(osg::Vec4 color);

	virtual void setTransparent(bool transparent);

	virtual void setRounding(float absRounding, float percentRounding);



	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);




protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;
	osg::ShapeDrawable* _sphere;
	static osg::Program* getOrLoadProgram();
	static osg::Program* _dialProg;

	osg::Vec4 _color;
	osg::Uniform* _absoluteRounding;
	osg::Uniform* _percentRounding;
	float _value;
	bool _jump;
	float leftPointX;
	float rightPointX;
	float topPointX;
	float height;
	float actualTop;

	bool holdingDial;
	osg::Matrix _startMat;
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};

class TentWindow : public cvr::UIElement, public UICallback, public UICallbackCaller

{
public:
	TentWindow();
	virtual void uiCallback(UICallbackCaller* ui);
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	void addTent(int index);
	void setTent(int index);
private:
	cvr::UIQuadElement* _bknd;
	std::unique_ptr<std::vector<Tent*>> _tents; 
	
	Tent* _tent;
	int _tentIndex;

	Dial* _dial;
	CallbackSlider* _bottomWidth;
	CallbackSlider* _centerPos;
	CallbackSlider* _topWidth;
	CallbackSlider* _height;
	CallbackSlider* _bottom;
	VolumeGroup* _volume;

	cvr::UIText* cVLabel;
	cvr::UIText* bVLabel;
	cvr::UIText* tVLabel;
	cvr::UIText* hVLabel;
	cvr::UIText* lVLabel;

	
};

enum organRGB { BLADDER, COLON, KIDNEY, SPLEEN, BODY};
class ColorPicker : public cvr::UIElement, public UICallback, public UICallbackCaller

{
public:
	ColorPicker();
	virtual void uiCallback(UICallbackCaller* ui);

	
	osg::Vec3 returnColor();
	void setButton(cvr::UIQuadElement* target);
	void setColorDisplay(ShaderQuad* colorDisplay) { _colorDisplay = colorDisplay; }
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	void setFunction(std::string transferFunction) { _transferFunction = transferFunction; }
	void setRadial(CallbackRadial* radial) { _radial = radial; }
	void setOrganRgb(organRGB organName) { _organRGB = organName;  }
	void setSaveColor(osg::Vec3* currOrgan) { _saveColor = currOrgan; }
	void setCPColor(osg::Vec3 savedColor) { 
		color = savedColor;
		_hue->setHue(color.x());
		_hue->setSV(osg::Vec2(color.y(), color.z()));
		_sv->setSV(osg::Vec2(color.y(), color.z()));
		_sv->setHue(color.x());
		
		_sv->set_indicator();
		_hue->set_indicator();
	}


	

private:
	ColorPickerHue* _hue;
	ColorPickerSaturationValue* _sv;
	cvr::UIQuadElement* _bknd;

	cvr::UIQuadElement* _target;//ga
	unsigned short int _currOrgan;
	osg::Vec3 color;

	//GA
	osg::Vec3 rgbColor;
	osg::Vec3* _saveColor;//HSV
	std::string _transferFunction;
	organRGB _organRGB;

	VolumeGroup* _volume;
	ShaderQuad* _colorDisplay;
	CallbackRadial* _radial;


	//GA

};





#endif