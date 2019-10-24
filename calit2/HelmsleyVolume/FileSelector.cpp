#include "FileSelector.h"
#include "HelmsleyVolume.h"

#ifdef WIN32
#include "dirent.h"
#else
#include <dirent.h>
#endif
#include <stdio.h>

#undef max
#include <algorithm>

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
				//Load volume
				if (_state == CHANGING)
				{
					HelmsleyVolume::instance()->removeVolume(0);
				}

				std::string maskpath = "";

				DIR* dir = opendir((_currentPath + "/mask").c_str());
				if (dir != NULL)
				{
					maskpath = _currentPath + "/mask";
					closedir(dir);
				}

				HelmsleyVolume::instance()->loadVolume(_currentPath, maskpath);
				_state = CHANGE;
				addVol->setText("Change Volume");
				volumeFileSelector->setVisible(false);
			}
			else
			{
				_currentPath = _currentPath + "/" + button->getText();
				updateFileSelection();
			}
		}
	}
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
