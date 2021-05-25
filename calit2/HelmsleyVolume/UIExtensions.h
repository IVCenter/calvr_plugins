#ifndef UI_EXTENSIONS_H
#define UI_EXTENSIONS_H

#define UI_BACKGROUND_COLOR osg::Vec4(0.15, 0.15, 0.15, 1)
#define UI_ACTIVE_COLOR osg::Vec4(.243, .561, 1, 1)
#define UI_RED_ACTIVE_COLOR osg::Vec4(1.0, 0.0, 0.0, 1)
#define UI_INACTIVE_COLOR osg::Vec4(.243, .561, 1, .4)
#define UI_BLUE_COLOR osg::Vec3(0.8, 0.9, 1.0)
#define UI_BLUE_COLOR2 osg::Vec4(.04, .25, .4, 1.0)
#define UI_RED_HOVER_COLOR osg::Vec3(1, 0.4, 0.4)
#define UI_RED_HOVER_COLOR_VEC4 osg::Vec4(1, 0.4, 0.4, 1)
#define UI_RED_DISABLED_COLOR_VEC4 osg::Vec4(0.51, 0.282, 0.2671, .4)
#define UI_NULL_COLOR_VEC4 osg::Vec4(-1, 0, 0, 0)
#define UI_YELLOW_COLOR osg::Vec3(1, 0.964, 0.8)
#define UI_GREEN_COLOR osg::Vec3(0.8, 1, 0.847)
#define UI_PURPLE_COLOR osg::Vec3(0.847, 0.8, 1)
#define UI_PINK_COLOR osg::Vec3(1, 0.8, 0.976)
#define UI_WHITE_COLOR osg::Vec4(1, 1, 1, 1)
#define UI_INACTIVE_WHITE_COLOR osg::Vec4(1, 1, 1, .4)
#define UI_INACTIVE_RED_COLOR osg::Vec4(0.8, 0.1, 0.1, 0.4)
#define UI_BLACK_COLOR osg::Vec4(0, 0, 0, 1)

#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UICheckbox.h>
#include <cvrMenu/NewUI/UIToggle.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/NewUI/UIText.h>
#include <cvrMenu/NewUI/UIRadial.h>
#include <cvrMenu/NewUI/UIList.h>
#include <cvrMenu/NewUI/UIQuadElement.h>

#include <cvrConfig/ConfigManager.h>

#include "VolumeGroup.h"

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
		_id = "";
	}

	void setCallback(UICallback* callback) { _callback = callback; }
	void setId(std::string id) { _id = id; }
	std::string getId() { return _id; }
protected:
	UICallback* _callback;
	std::string _id;
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

class HoverButton
	: public CallbackButton
{
public:
	HoverButton(cvr::UIQuadElement* bknd, osg::Vec4 originalColor, osg::Vec4 hoverColor) :
		_bknd(bknd), _originalColor(originalColor), _hoverColor(hoverColor) {}

	void processHover(bool enter) override
	{
		if (_disabledColor == UI_NULL_COLOR_VEC4) {		//If the button is not disabled...
			if (enter)
			{
				_bknd->setColor(_hoverColor);
			}
			else {
				_bknd->setColor(_originalColor);
			}
		}
	}

	void setDisabledColor(osg::Vec4 color) 
	{ 
		_disabledColor = color;
		if(_disabledColor != UI_NULL_COLOR_VEC4)
			_bknd->setColor(_disabledColor);
		else {
			_bknd->setColor(_originalColor);
		}
	}

	cvr::UIQuadElement* _bknd;
	osg::Vec4 _originalColor;
	osg::Vec4 _hoverColor;
	osg::Vec4 _disabledColor = UI_NULL_COLOR_VEC4;
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
	void setColor(osg::Vec4 color);
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
		_uniforms = std::map<std::string, osg::ref_ptr<osg::Uniform>>();
	}
	~ShaderQuad(); 

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
	std::map<std::string, osg::ref_ptr<osg::Uniform>> _uniforms;
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
 class TriangleButton : public cvr::UIElement, public UICallbackCaller
{
public:
	TriangleButton(osg::Vec4 color = osg::Vec4(UI_BLUE_COLOR, 1), osg::Vec4 hoverColor = osg::Vec4(UI_BLUE_COLOR,1))
		: UIElement(), _hoverColor(hoverColor)
	{
		_color = color;
		_geode = new osg::Geode();
		createGeometry();

		_colorUniform = new osg::Uniform("Color", _color);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_colorUniform);

		setProgram(getOrLoadProgram());
	}

	virtual void createGeometry();
	virtual void updateGeometry();
	void setRotate(double radians);



	virtual void setColor(osg::Vec3 color);

	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);
	virtual bool processEvent(cvr::InteractionEvent* event) override;
	virtual void processHover(bool enter) override {
		osg::Vec4 temp = _color;
		_color = _hoverColor;
		_hoverColor = temp;
		_colorUniform->set(_color);

		_dirty = true;
	}
	osg::Uniform* _colorUniform;
	osg::ref_ptr<osg::Geode> _geode;
protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;
	osg::Matrix _rot;

	static osg::Program* getOrLoadProgram(); 
	static osg::Program* _triangleButtonProg;

	osg::Vec4 _color;
	
	osg::Vec4 _hoverColor;

	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};

class Tent : public cvr::UIElement
{
public:
	Tent(osg::Vec4 color = osg::Vec4(UI_BLUE_COLOR, 1))
		: UIElement()
	{
		leftPointX = -.5;
		rightPointX = .5;
		topPointX = 0.5;
		actualTop = 0.0;
		height = 0.0;
		bottomHeight = -1.0;
		actualBottomHeight = -1.0;
		savedHeight = -1.0;
		_center = .5;
		_color = color;
		_geode = new osg::Geode();

		createGeometry();
	
		_absoluteRounding = new osg::Uniform("absoluteRounding", 0.0f);
		_percentRounding = new osg::Uniform("percentRounding", 0.0f);
		_centerUniform = new osg::Uniform("Center", 0.5f);
		_widthUniform = new osg::Uniform("Width", 0.5f);
		_colorUniform = new osg::Uniform("Color", UI_BLUE_COLOR);
		_selectedUniform = new osg::Uniform("Selected", true);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_absoluteRounding);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_percentRounding);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_centerUniform);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_widthUniform);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_colorUniform);
		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_selectedUniform);

		this->setTransparent(true);
		
		
		setProgram(getOrLoadProgram());
		addUniform("SV", osg::Vec2(1.0f, 1.0f));
	}

	float changeBottomVertices(float x);
	float changeTopVertices(float x);
	float changeHeight(float x);
	float changeBottomHeight(float x);
	void setCenter(float x) { _center = x; _centerUniform->set(x);}
	void setSavedHeight(float x) { savedHeight = x - 1.0; }
	void setSelected(bool selected) { _selectedUniform->set(selected); }
	float getBottomWidth() {return rightPointX;}
	float getTopWidth() {return topPointX;}
	float getCenter() {return _center;}
	float getHeight() {return height + 1.0;}
	float getBottom() {return bottomHeight + 1.0;}
	float getSavedHeight() {return savedHeight + 1.0;}
	
	virtual void createGeometry();
	virtual void updateGeometry();




	virtual void setColor(osg::Vec3 color);

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

	osg::Uniform* _absoluteRounding;
	osg::Uniform* _percentRounding;
	osg::Uniform* _centerUniform;
	osg::Uniform* _widthUniform;
	osg::Uniform* _colorUniform;
	osg::Uniform* _selectedUniform;


protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;

	static osg::Program* getOrLoadProgram();
	static osg::Program* _triangleProg;
 	osg::Vec4 _color;
	

	float leftPointX;
	float rightPointX;
	float topPointX;
	float height;
	float savedHeight;
	float actualTop; 
	float actualBottomHeight; 
	float bottomHeight;
	float _center;
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};
//curved quad//////////////
class CurvedQuad : public CallbackToggle
{
public:
	CurvedQuad(int curr = 1, int total = 1, osg::Vec4 color = osg::Vec4(UI_BLUE_COLOR, 1))
		: CallbackToggle()
	{
		 
		_color = color;
		_geode = new osg::Geode();

		_curr = curr;
		_total = total;
		
		// add the stateset tor the drawable
		

		createGeometry();
	
   		_colorUniform = new osg::Uniform("Color", color);
 		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_colorUniform);
 
		this->setTransparent(true);
		
		
		setProgram(getOrLoadProgram());
 	}

 	
	virtual void createGeometry();
	virtual void updateGeometry();
 	virtual void setColor(osg::Vec4 color);
 	virtual void setTransparent(bool transparent);
 
 
	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }
	virtual void processHover(bool enter) override;


	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);

   	osg::Uniform* _colorUniform;
 
	static osg::Program* _curvedQuadProg;
	


protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;

	static osg::Program* getOrLoadProgram();
	static osg::Program* _triangleProg;

	osg::Vec4 _color;
	
	int _curr = 1;	//For Segmenting
	int _total = 1;
	
	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};
 
class Line : public cvr::UIElement
{
public:
	Line(osg::Vec3dArray* coords, osg::Vec4 color = osg::Vec4(UI_RED_HOVER_COLOR, 1))
		: UIElement()
	{
		_coords = coords;
		_color = color;
		_geode = new osg::Geode();
		
		createGeometry();

		
		_colorUniform = new osg::Uniform("Color", UI_RED_HOVER_COLOR);

		(_geode->getDrawable(0))->getOrCreateStateSet()->addUniform(_colorUniform);

		this->setTransparent(true);


		setProgram(getOrLoadProgram());

	}

	virtual void createGeometry();
	virtual void updateGeometry();
	virtual void setColor(osg::Vec3 color);
	virtual void setTransparent(bool transparent);

	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	virtual osg::Program* getProgram() { return _program; }
	virtual osg::Geode* getGeode() { return _geode; }

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);

	osg::Uniform* _colorUniform;
	osg::ref_ptr<osg::MatrixTransform> getTransform() {
		return _transform;
	}

protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3dArray* _coords;

	static osg::Program* getOrLoadProgram();
	static osg::Program* _lineProg;

	osg::Vec4 _color;

	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;

};

class MarchingCubesRender : public cvr::UIElement
{
public:
	MarchingCubesRender(osg::ref_ptr<osg::Vec3Array> coords, osg::Vec3i volDims, osg::ref_ptr<osg::Geometry> geom, unsigned int verticeCount, osg::ref_ptr<osg::Vec3Array> va
,osg::ref_ptr<osg::ShaderStorageBufferBinding> ssbb)
		: UIElement(), _verticeCount(verticeCount), _VA(va), _ssbb(ssbb)
	{
		_coords = coords;
	
		_geode = new osg::Geode();
		_voldims = volDims;

		_mcGeom = geom;
		
		setProgram(getOrLoadProgram());
		createGeometry();
	}

	virtual void createGeometry();
	virtual void updateGeometry();

	virtual void setProgram(osg::Program* p) { _program = p; _dirty = true; }
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	virtual osg::Program* getProgram() { return _program; }
	virtual const osg::ref_ptr<osg::Geode> getGeode() { return _geode; }
	virtual bool processEvent(cvr::InteractionEvent* event) override;

	template <typename T>
	void addUniform(std::string uniform, T initialvalue);
	void addUniform(std::string uniform);
	virtual void addUniform(osg::Uniform* uniform);
	virtual osg::Uniform* getUniform(std::string uniform);
	virtual void setShaderDefine(std::string name, std::string definition, osg::StateAttribute::Values on);

	void printSTLFile();

 	osg::ref_ptr<osg::MatrixTransform> getTransform() {
		return _transform;
	}

	osg::ref_ptr<osg::ShaderStorageBufferBinding> _ssbb;

protected:
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::ref_ptr<osg::Vec3Array> _coords;
	osg::ref_ptr<osg::Vec3Array> _VA;
	osg::Vec3i _voldims;
	VolumeGroup* _volume;


	static osg::Program* getOrLoadProgram();
	static osg::Program* _mcProg;

	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;


	osg::ref_ptr<osg::Geometry> _mcGeom;
	unsigned int _verticeCount;
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



class TentWindowOnly : public cvr::UIElement, public UICallback, public UICallbackCaller

{
public:
	TentWindowOnly();
	virtual void uiCallback(UICallbackCaller* ui);
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	Tent* addTent(int index, osg::Vec3 color);
	void setTent(int index);
	void toggleTent(int index);
	void clearTents();
	void setUpGradients();
	void fillTentDetails(int _triangleIndex, float center, float bottomWidth, float topWidth, float height, float lowest);
	std::vector<float> getPresetData(int index);
	std::vector<Tent*> _tents;
	cvr::UIQuadElement* _bknd;
private:
	

	ShaderQuad* _colorGrad;
	ShaderQuad* _opacGrad;
	ShaderQuad* _opacColorGrad;

	Tent* _tent;
	int _tentIndex;

	VolumeGroup* _volume;

};
////////////Hist
class HistQuad : public cvr::UIElement

{
public:
	HistQuad();
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	void setBB(osg::ref_ptr<osg::ShaderStorageBufferBinding> bb);
	void setMax(unsigned int histMax);

	ShaderQuad* _bknd;
	static osg::Program* getOrLoadProgram();
	static osg::Program* _histprogram;

	osg::ref_ptr<osg::ShaderStorageBufferBinding> histBB = nullptr;
private:
	


	VolumeGroup* _volume;

};
////////////Hist

class TentWindow : public cvr::UIElement, public UICallback, public UICallbackCaller

{
public:
	TentWindow(TentWindowOnly* tWOnly);
	virtual void uiCallback(UICallbackCaller* ui);
	void setVolume(VolumeGroup* volume) { _volume = volume; }
	Tent* addTent(int index, osg::Vec3 color);
	void initDials();
	void setDialList(cvr::UIList* list);
	void setTent(int index);
	void toggleTent(int index);
	void clearTents();
	void fillTentDetails(int _triangleIndex, float center, float bottomWidth, float topWidth, float height, float lowest);
	std::vector<float> getPresetData(int index);
	std::vector<float> getCinePresetData(int index);
	//std::unique_ptr<std::vector<Tent*>> _tents;
	TentWindowOnly* _tWOnly;
private:
	cvr::UIQuadElement* _bknd;

	
	Tent* _tent;
	//TentWindowOnly* _tWOnly;
	int _tentIndex;

	Dial* _dialBW;
	Dial* _dialCenter;
	Dial* _dialTW;
	Dial* _dialHeight;
	Dial* _dialBottom;
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
	osg::Vec3 RGBtoHSV(float fR, float fG, float fB);
	void setSaveColor(osg::Vec3* currOrgan) { _saveColor = currOrgan; }

	void setCPColor(osg::Vec3 savedColor) { 
		color = RGBtoHSV(savedColor.x(), savedColor.y(), savedColor.z());

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

	cvr::UIQuadElement* _target;
	unsigned short int _currOrgan;
	osg::Vec3 color;

	
	osg::Vec3 rgbColor;
	osg::Vec3* _saveColor;
	std::string _transferFunction;
	organRGB _organRGB;

	VolumeGroup* _volume;
	ShaderQuad* _colorDisplay;
	CallbackRadial* _radial;

};


class ColorSlider : public cvr::UIElement, public UICallback, public UICallbackCaller
{
public:
	ColorSlider(ColorPicker* cp, osg::Vec4 color = osg::Vec4(1, 1, 1, 1))
		: UIElement()
	{
		_color = color;
		_geode = new osg::Geode();
		createGeometry();


		setProgram(getOrLoadProgram());

	}
	virtual void createGeometry();
	virtual void updateGeometry();

	virtual bool processEvent(cvr::InteractionEvent* event) override;

	virtual void setColor(osg::Vec4 color);
	osg::Vec4 getColor() { return _color; }
	virtual void uiCallback(UICallbackCaller* ui);



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
	ColorPicker* _cp;
	osg::ref_ptr<osg::MatrixTransform> _transform;
	osg::ref_ptr<osg::Geode> _geode;
	osg::Geometry* _polyGeom;
	osg::Vec3* _coords;
	osg::ShapeDrawable* _sphere;
	static osg::Program* getOrLoadProgram();
	static osg::Program* _colorSlideProg;

	osg::Vec4 _color;

	osg::ref_ptr<osg::Program> _program;
	std::map<std::string, osg::Uniform*> _uniforms;



};

class Selection : public cvr::UIElement, public UICallback, public UICallbackCaller {
public:
	Selection(std::string name, bool hasMask = false);

	virtual void uiCallback(UICallbackCaller* ui);

	virtual bool processEvent(cvr::InteractionEvent* event) override;

	void setButtonCallback(UICallback* ui) { _button->setCallback(ui); }

	std::string getName() { return _name; }
	CallbackButton* getButton() { return _button; }

	void setName(std::string name);
	void setMask(bool hasMask);
	void lowerTextSize();
	void setImage(cvr::UITexture* uiTexture);
	cvr::UITexture* getImage() { return _uiTexture; }
	void removeImage();
protected:
	void addMaskSymbol();
	void removeMaskSymbol();

	cvr::UIQuadElement* _bknd;
	CallbackButton* _button;
	std::string _name;
	cvr::UIText* _uiText;
	cvr::UITexture* _uiTexture;
};

class FullButton: public cvr::UIElement, public UICallback, public UICallbackCaller {
public:
	FullButton(std::string txt, float textsize = 50.f, osg::Vec4 color = UI_RED_ACTIVE_COLOR, osg::Vec4 color2 = UI_NULL_COLOR_VEC4);

	virtual void uiCallback(UICallbackCaller* ui);



	virtual bool processEvent(cvr::InteractionEvent* event) override;
	virtual void processHover(bool enter) override;

	void setButtonCallback(UICallback* ui) { _button->setCallback(ui); }

	std::string getName() { return _name; }
	HoverButton* getButton() { return _button; }
	cvr::UIText* getText() { return _uiText; }

protected:
	

	cvr::UIQuadElement* _bknd;
	HoverButton* _button;
	std::string _name;
	cvr::UIText* _uiText;
	osg::Vec4 _originalColor;
	osg::Vec4 _savedColor;
 };

#endif