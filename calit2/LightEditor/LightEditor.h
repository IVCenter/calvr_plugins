#ifndef _LIGHT_EDITOR_H_
#define _LIGHT_EDITOR_H_

#include <iostream>
#include <list>

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuImage.h>
#include <cvrMenu/MenuList.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuText.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/PopupMenu.h>

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
        bool processEvent(cvr::InteractionEvent * event);

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
