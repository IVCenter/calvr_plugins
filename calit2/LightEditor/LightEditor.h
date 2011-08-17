#ifndef _LIGHT_EDITOR_H_
#define _LIGHT_EDITOR_H_

#include <iostream>
#include <list>

#include <config/ConfigManager.h>
#include <kernel/CVRPlugin.h>
#include <menu/MenuButton.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuImage.h>
#include <menu/MenuList.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuText.h>
#include <menu/SubMenu.h>
#include <menu/PopupMenu.h>

#include "LightManager.h"
#include "LightShading.h"

class LightEditor : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        LightEditor();
        ~LightEditor();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        void preFrame();
        bool buttonEvent(int type, int button, int hand, const osg::Matrix& mat);
        bool mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix& mat);

    protected:
        LightManager * mLightManager;   
        LightShading * mLightShading;   

        // Menu objects
        cvr::SubMenu *           _lightMenu;
        cvr::MenuButton *        _createNewLightButton;
        cvr::MenuText *          _selectedLightText;
        cvr::MenuList *          _selectLightList;
        cvr::MenuCheckbox *      _graphicModelsCheckbox;
        cvr::MenuButton *        _saveLightsButton;

        // + Edit Light Menu
        cvr::PopupMenu *         _elPopup;

        cvr::MenuCheckbox *      _elToggleEnable;

        cvr::MenuText *          _elLightTypeText;
        cvr::MenuList *          _elLightTypeList;

        cvr::MenuText *          _elColorTypeText;
        cvr::MenuList *          _elColorTypeList;
        
        cvr::MenuRangeValue *    _elR;
        cvr::MenuRangeValue *    _elG;
        cvr::MenuRangeValue *    _elB; 

        cvr::MenuText *          _elAttenuationText;	
        cvr::MenuList *          _elAttenuationList;
        cvr::MenuRangeValue *    _elAttenuation; 

        cvr::MenuText *	         _elLabelSpotDirection;		
        cvr::MenuRangeValue *    _elSpotExponent; 
        cvr::MenuRangeValue *    _elSpotCutoff; 

        // End Menu Objects

        void initEditLightMenu();
        void key(int type, int keySym, int mod);
        void updateEditLightMenu();

        void addNewLight();
        void repopulateSelectLightMenu();
};

#endif
