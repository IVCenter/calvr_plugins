#ifndef UI_EXTENSIONS_H
#define UI_EXTENSIONS_H

#include <cvrMenu/NewUI/UIButton.h>
#include <cvrMenu/NewUI/UICheckbox.h>
#include <cvrMenu/NewUI/UIToggle.h>
#include <cvrMenu/NewUI/UISlider.h>
#include <cvrMenu/NewUI/UIText.h>

#include <cvrConfig/ConfigManager.h>

//Callback class (NewUI doesnt implement callback by default - extended classes will for now just to make things easier
class UICallback
{
public:
	UICallback() {}

	virtual void uiCallback(cvr::UIElement* e) = 0;
};

class VisibilityToggle : public cvr::UIToggle
{
public:

	VisibilityToggle(std::string text);

	virtual bool onToggle() override;

	void setCallback(UICallback* callback) { _callback = callback; }

	cvr::UICheckbox* eye;
	cvr::UIText* label;

protected:
	UICallback* _callback;
};

#endif