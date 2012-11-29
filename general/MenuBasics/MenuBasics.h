#ifndef CALVR_MENU_BASICS
#define CALVR_MENU_BASICS

#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValueCompact.h>
#include <cvrKernel/CVRPlugin.h>

class MenuBasics : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        MenuBasics();
        ~MenuBasics();

        bool init();

        virtual int getPriority() { return 95; }

        void menuCallback(cvr::MenuItem * item);

        void preFrame();

        void message(int type, char * & data, bool);
    protected:
        cvr::MenuCheckbox * moveworld;
        cvr::MenuCheckbox * scale;
        cvr::MenuCheckbox * drive;
        cvr::MenuCheckbox * fly;
        cvr::MenuCheckbox * snap;
        cvr::MenuRangeValueCompact * navScale;

        cvr::MenuCheckbox * activeMode;

        cvr::MenuCheckbox * stopHeadTracking;
        cvr::MenuCheckbox * eyeSeparation;
        cvr::MenuCheckbox * omniStereo;
        bool changeSep;
        float sepStep;

        cvr::MenuButton * viewall;
        cvr::MenuButton * resetview;

        float sceneSize;
};

#endif
