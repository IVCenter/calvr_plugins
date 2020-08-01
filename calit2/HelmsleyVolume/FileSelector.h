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

	enum SelectChoice {
		LEFT,
		RIGHT,
	};

	static std::vector<std::string> getPresets();
	virtual void uiCallback(UICallbackCaller* ui);
	std::string getCurrPath() { return _currentPath; }

	void loadVolume(std::string seriesPath, bool change, bool onlyVolume);
	void loadVolumeOnly(std::string seriesPath);
	
protected:
	void updateSelections(SelectChoice choice);
	void updateFileSelection(); //OLD
	//void newUpdateFileSelection();
	int checkIfPatient(std::string fn, int indexFromDicom);
	bool checkSelectionCallbacks(UICallbackCaller* item);
	void loadPatient(std::string pName);
	int loadSeriesList(std::string pFN, int indexFromDicom);
	void showDicomThumbnail();
	std::string getMiddleImage(int seriesIndex);

	cvr::SceneObject* _so;

	cvr::PopupMenu* addVolumeMenu;
	cvr::MenuButton* addVol;
	cvr::PopupMenu* volumeFileSelector;
	
	cvr::UIList* _topList;
	cvr::UIList* _botList;
	TriangleButton* _rightArrow;
	TriangleButton* _leftArrow;
	int _selectIndex;
	std::vector<Selection*> _selections;
	
	//std::map<cvr::MenuItem*, std::string> loadVolumeMenu;
	std::vector<cvr::MenuItem*> pathSelections;
	std::map<std::string, std::string> _patientDirectories;	//Patient Folder Name, Directory
	std::map<std::string, std::string> _seriesList;
	std::map<std::string, std::string>* _currMap;
	MenuState _state;
	std::string _currentPath;
};



#endif
