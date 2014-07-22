#ifndef SMV2SETTINGS
#define SMV2SETTINGS

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/Screens/ScreenMVSimulator.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/PopupMenu.h>

class SMV2Settings : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        SMV2Settings();
        ~SMV2Settings();

        bool init();

        void menuCallback(cvr::MenuItem * item);

    protected:
        cvr::SubMenu * mvsMenu;
        cvr::MenuCheckbox * multipleUsers;
        cvr::SubMenu * contributionMenu;
        cvr::MenuCheckbox * orientation3d;
        cvr::MenuCheckbox * linearFunc;
        cvr::MenuCheckbox * cosineFunc;
        cvr::MenuCheckbox * gaussianFunc;
        cvr::MenuCheckbox * autoContrVar;
        cvr::MenuRangeValue * contributionVar;
        cvr::SubMenu * zoneMenu;
        cvr::MenuCheckbox * autoAdjust;
        cvr::MenuRangeValue * zoneRowQuantity;
        cvr::MenuRangeValue * zoneColumnQuantity;
        cvr::MenuRangeValue * autoAdjustTarget;
        cvr::MenuRangeValue * autoAdjustOffset;
        cvr::MenuCheckbox * zoneColoring;
};

#endif
