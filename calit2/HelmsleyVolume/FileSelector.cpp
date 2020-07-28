#pragma once
#include "FileSelector.h"
#include "HelmsleyVolume.h"
#include "Utils.h"
#include "UIExtensions.h"

#ifdef WIN32
#include "dirent.h"
#else
#include <dirent.h>
#endif
#include <stdio.h>

#undef max
#include <algorithm>

#define SLOTCOUNT 8
void FileSelector::init()
{
	_state = NEW;
	pathSelections = std::vector<cvr::MenuItem*>();

	addVolumeMenu = new cvr::PopupMenu("Volumes", "", false, true);
	osg::Vec3 menupos = cvr::ConfigManager::getVec3("Plugin.HelmsleyVolume.Orientation.FileMenu.Position", osg::Vec3(-600, 500, 1100));
	addVolumeMenu->setPosition(menupos);
	addVolumeMenu->setVisible(true);
	//addVolumeMenu->setMovable(false);

	addVol = new cvr::MenuButton("New Volume", true, "checkbox=TRUE.rgb");
	addVolumeMenu->addMenuItem(addVol);
	addVol->setCallback(this);

	volumeFileSelector = new cvr::PopupMenu("Choose File", "", false, true);
	volumeFileSelector->setPosition(menupos + osg::Vec3(0,0,-100));
	//volumeFileSelector->getRootSubMenu()->setTextScale(0.5);
	_currentPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.BaseFolder", "C:/", false);
	std::cout << "Volume search path: " << _currentPath << std::endl;





	////////////New FileSelect Implement.///////////
	_so = new cvr::SceneObject("FileSelect", false, true, false, false, false);
	cvr::PluginHelper::registerSceneObject(_so, "HelmsleyVolume");
	osg::Quat rot;
	rot.makeRotate(3.14/2, 0, 0, 1);
	_so->setRotation(rot);
	_so->setPosition(osg::Vec3(-1900, -1600, 2200));
	
	cvr::UIPopup* fsPopup = new cvr::UIPopup();

	fsPopup->getRootElement()->setAbsoluteSize(osg::Vec3(2400, 1, 1500));
	cvr::UIQuadElement* fsBknd = new cvr::UIQuadElement(UI_BACKGROUND_COLOR);
	_rightArrow = new TriangleButton(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	_rightArrow->setCallback(this);
	fsBknd->addChild(_rightArrow);
	_rightArrow->setPercentSize(osg::Vec3(0.07, 1.0, .2));
	_rightArrow->setPercentPos(osg::Vec3(0.95, -1.0, -.5));
	//_rightArrow->setPercentPos(osg::Vec3(0.8, 1.0, -.55));
	fsPopup->addChild(fsBknd);
	fsPopup->setActive(true, true);
	
	
	cvr::UIQuadElement* titleBknd = new cvr::UIQuadElement();
	fsPopup->addChild(titleBknd);
	titleBknd->setPercentPos(osg::Vec3(.4, -1, -.03));
	titleBknd->setPercentSize(osg::Vec3(.2, 1, .1));
	titleBknd->setBorderSize(.01);
	cvr::UIText* titleText = new cvr::UIText("Load Volume", 70.f, osgText::TextBase::CENTER_CENTER);
	titleText->setColor(osg::Vec4(0.0, 0.0, 0.0, 1.0));
	titleBknd->addChild(titleText);

	cvr::UIQuadElement* contentBknd = new cvr::UIQuadElement();
	contentBknd->setPercentSize(osg::Vec3(.8, 1, .8));
	contentBknd->setPercentPos(osg::Vec3(.1, -1, -.15));
	contentBknd->setBorderSize(.01);
	fsPopup->addChild(contentBknd);

	_so->addChild(fsPopup->getRoot());
	_so->attachToScene();
	fsPopup->getRootElement()->updateElement(osg::Vec3(0, 0, 0), osg::Vec3(0, 0, 0));
	_so->dirtyBounds();

	checkIfPatient(_currentPath, -1);

	


	_topList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	_botList = new cvr::UIList(cvr::UIList::LEFT_TO_RIGHT, cvr::UIList::CONTINUE);
	contentBknd->addChild(_topList);
	_topList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_topList->setPercentPos(osg::Vec3(0.1, 1, -.05));
	contentBknd->addChild(_botList);
	_botList->setPercentSize(osg::Vec3(0.8, 1, .4));
	_botList->setPercentPos(osg::Vec3(0.1, 1, -.55));

	int selectCount = 0;
	for (auto iter = _patientDirectories.begin(); iter != _patientDirectories.end(); ++iter)
	{
		std::string patient = iter->first;
		std::cout << patient << std::endl;

		Selection* selec = new Selection(iter->first);
		selec->setId("selection");
		selec->setButtonCallback(this);
		_selections.push_back(selec);
		selectCount < 4 ? _topList->addChild(selec) : _botList->addChild(selec);
		selectCount++;


		if (selectCount == 8) {
			_selectIndex = selectCount;	
			break;
		}

	}
}

void FileSelector::menuCallback(cvr::MenuItem* item)
{
	if (_state == NEW || _state == CHANGE)
	{
		if (item == addVol)
		{
			if (_state == NEW)
			{
				_state = CHOOSING;
			}
			else
			{
				_state = CHANGING;
			}
			volumeFileSelector->setVisible(true);
			updateFileSelection();
		}
	}
	else if (_state == CHOOSING || _state == CHANGING)
	{
		if (std::find(pathSelections.begin(), pathSelections.end(), item) != pathSelections.end())
		{
			cvr::MenuButton* button = (cvr::MenuButton*)item;
			if (strcmp(button->getText().c_str(), "..") == 0)
			{
				//Go back a folder
				_currentPath = _currentPath.substr(0, _currentPath.find_last_of("\\/"));
				updateFileSelection();
			}
			else if (button->getText().find(".dcm") != std::string::npos && 
				button->getText().find(".dcm") == button->getText().find_last_of("."))
			{
				////Load volume
				bool change = _state == CHANGING ? true : false;
				loadVolume(_currentPath, change, false);
			}
			else
			{
				_currentPath = _currentPath + "/" + button->getText();
				updateFileSelection();
			}
		}
	}
}

void FileSelector::uiCallback(UICallbackCaller* ui){
	if (ui == _rightArrow) {
		std::cout << "within fileselect" << std::endl;
		if(_patientDirectories.size() > _selectIndex)
			updateSelections();
	}
	//std::cout << _patientDirectories[ui->getId()];
	//newUpdateFileSelection();
}

void FileSelector::updateSelections() {
	int slotsLeft = 8;
	int topBotIndex = 0;

	auto it = _patientDirectories.begin();
	std::advance(it, _selectIndex);
	
	while (slotsLeft > 0) {
		
		Selection* selec;
		topBotIndex < 4 ? selec = (Selection*)_topList->getChild(topBotIndex)
						 : selec = (Selection*)_botList->getChild(topBotIndex % 4);
		
		_patientDirectories.size() > _selectIndex ? selec->setName(it->first)
												  : selec->setName("");
		topBotIndex++;
		it++;			//path map iterator
		_selectIndex++; //path map "index"
		slotsLeft--;	
	}
	
}

void FileSelector::loadVolume(std::string seriesPath, bool change, bool onlyVolume) {
	//Load volume
	if (change)
	{
		HelmsleyVolume::instance()->removeVolume(0, onlyVolume);
	}

	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolume(seriesPath, maskpath, onlyVolume);
	_state = CHANGE;
	addVol->setText("Change Volume");
	volumeFileSelector->setVisible(false);
}

void FileSelector::loadVolumeOnly(std::string seriesPath) {
	HelmsleyVolume::instance()->removeVolumeOnly(0);
	std::string maskpath = "";

	DIR* dir = opendir((seriesPath + "/mask").c_str());
	if (dir != NULL)
	{
		maskpath = seriesPath + "/mask";
		closedir(dir);
	}

	HelmsleyVolume::instance()->loadVolumeOnly(seriesPath, maskpath);
	_state = CHANGE;
	addVol->setText("Change Volume");
	volumeFileSelector->setVisible(false);
}

void FileSelector::updateFileSelection()
{
	//Clear current directory options
	for (int i = 0; i < pathSelections.size(); ++i)
	{
		volumeFileSelector->removeMenuItem(pathSelections[i]);
		//delete(pathSelections[i]);
	}
	pathSelections.clear();

	int begin = std::max((int)(_currentPath.size() - 25), 0);
	std::string title = "";
	if (begin > 0)
	{
		title += "...";
	}
	title += _currentPath.substr(begin, _currentPath.size());
	volumeFileSelector->setTitle(title);


	DIR* dir = opendir(_currentPath.c_str());

	int dcmCount = 0;
	cvr::MenuButton* dcmButton = NULL;
	
	struct dirent* entry = readdir(dir);
	while (entry != NULL)
	{
		if ((entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "mask") != 0) || //folders
			(entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0)) //first dcm file
		{
			if (entry->d_type == DT_REG)
			{
				dcmCount++;
				if (dcmCount > 1)
				{
					entry = readdir(dir);
					continue;
				}
			}
			cvr::MenuButton* button = new cvr::MenuButton(entry->d_name);
			if (entry->d_type == DT_REG)
			{
				dcmButton = button;
			}
			button->setCallback(this);
			pathSelections.push_back(button);
			volumeFileSelector->addMenuItem(button);
		}


		entry = readdir(dir);
	}

	if (dcmButton)
	{
		dcmButton->setText(dcmButton->getText() + "    (" + std::to_string(dcmCount) + " files)");
	}

	closedir(dir);
}

int FileSelector::checkIfPatient(std::string fn, int indexFromDicom) {
	//for every file/direct check recursively if there is a series to dicom connection
	//if true mark file/direct as patient and add to menu
	

	DIR* dir = opendir(fn.c_str());
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
			std::cout << "fn: " << fn << std::endl;
			std::cout << "newfn: " << entry->d_name << std::endl;
			std::string newFn = fn + "/" + entry->d_name;
			int newIndex = checkIfPatient(newFn, indexFromDicom);
			if (newIndex > -1) {
				if (newIndex == 1) {
					_patientDirectories[std::string(entry->d_name)] = fn+"/"+entry->d_name;
				}
				else {
					closedir(dir);
					return newIndex + 1;
				}
			}
		}
		else if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "dcm") == 0) {
			closedir(dir);
			return indexFromDicom + 1;
		}
		entry = readdir(dir);
	}
	closedir(dir);
	return indexFromDicom + 0;

}

std::vector<std::string> FileSelector::getPresets() {
	std::string currPath = cvr::ConfigManager::getEntry("Plugin.HelmsleyVolume.PresetsFolder", "C:/", false);
	std::string presetPath = currPath;
	DIR* dir = opendir(presetPath.c_str());
	if (dir == nullptr) {
		std::cout << "Directory not found in: " << presetPath << std::endl;
	}
	std::vector<std::string> presetPaths;
	struct dirent* entry = readdir(dir);
	while (entry != NULL) {
		if (entry->d_type == DT_REG && strcmp(strrchr(entry->d_name, '.') + 1, "yml") == 0) {
			presetPaths.push_back(presetPath + "\\" + entry->d_name);
		}
		entry = readdir(dir);
	}
	return presetPaths;
}

