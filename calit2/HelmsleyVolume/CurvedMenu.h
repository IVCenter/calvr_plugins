#pragma once
#include "UIExtensions.h"


class CurvedMenu : public cvr::UIElement {
public:
	CurvedMenu(UICallback* callback, int numItems = 8);

	void setImage(int index, std::string iconPath);

protected:
	int _numItems = 0;
	UIElement* _parent;
	cvr::UIList* _list;
};