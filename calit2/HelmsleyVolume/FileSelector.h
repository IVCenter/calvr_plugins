#pragma once
#ifndef FILE_SELECTOR_H
#define FILE_SELECTOR_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>

#include <cvrMenu/MenuItem.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/PopupMenu.h>

#include <cvrConfig/ConfigManager.h>

class FileSelector : public cvr::MenuCallback {
public:
	FileSelector() 
	{
		init();
	}

	void init();

	virtual void menuCallback(cvr::MenuItem * item);

	enum MenuState {
		NEW,
		CHOOSING,
		CHANGE,
		CHANGING
	};

protected:

	void updateFileSelection();

	cvr::PopupMenu* addVolumeMenu;
	cvr::MenuButton* addVol;
	cvr::PopupMenu* volumeFileSelector;
	
	//std::map<cvr::MenuItem*, std::string> loadVolumeMenu;
	std::vector<cvr::MenuItem*> pathSelections;

	MenuState _state;
	std::string _currentPath;
};

#endif
