#ifndef UI_EXTENSIONS_H
#define UI_EXTENSIONS_H

#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UICheckbox.h>
#include <cvrMenu/NewUI/UIToggle.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/NewUI/UIText.h>
#include <cvrMenu/NewUI/UIRadial.h>
#include <cvrMenu/NewUI/UIList.h>

#include <cvrConfig/ConfigManager.h>

//Callback class (NewUI doesnt implement callback by default - extended classes will for now just to make things easier
class UICallback
{
public:
	UICallback() {}

	virtual void uiCallback(cvr::UIElement* e) = 0;
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

class ToolRadial : public cvr::UIRadial, public UICallbackCaller
{
public:
	ToolRadial();

	virtual void onSelectionChange() override;
};

class ToolSelector : public cvr::UIList
{
public:
	ToolSelector(Direction d = LEFT_TO_RIGHT, OverflowBehavior o = CUT);

protected:
	ToolRadial* _toolRadial;
};

#endif