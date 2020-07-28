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
#include "UIExtensions.h"
#include <cvrConfig/ConfigManager.h>

class FileSelector : public cvr::MenuCallback, public UICallback, public UICallbackCaller {
public:
	FileSelector() 
	{
		init();
	}
	//TODO Deconstructor
	void init();

	virtual void menuCallback(cvr::MenuItem * item);

	enum MenuState {
		NEW,
		CHOOSING,
		CHANGE,
		CHANGING
	};

	static std::vector<std::string> getPresets();
	virtual void uiCallback(UICallbackCaller* ui);
	std::string getCurrPath() { return _currentPath; }

	void loadVolume(std::string seriesPath, bool change, bool onlyVolume);
	void loadVolumeOnly(std::string seriesPath);
	
protected:
	void updateSelections();
	void updateFileSelection(); //OLD
	//void newUpdateFileSelection();
	int checkIfPatient(std::string fn, int indexFromDicom);

	cvr::SceneObject* _so;

	cvr::PopupMenu* addVolumeMenu;
	cvr::MenuButton* addVol;
	cvr::PopupMenu* volumeFileSelector;
	
	cvr::UIList* _topList;
	cvr::UIList* _botList;
	TriangleButton* _rightArrow;
	int _selectIndex;
	std::vector<Selection*> _selections;
	//std::map<cvr::MenuItem*, std::string> loadVolumeMenu;
	std::vector<cvr::MenuItem*> pathSelections;
	std::map<std::string, std::string> _patientDirectories;	//Patient Folder Name, Directory
	MenuState _state;
	std::string _currentPath;
};



#endif
