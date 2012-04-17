#ifndef _TouchDesigner_
#define _TouchDesigner_

#include <kernel/CVRPlugin.h>
#include <kernel/SceneObject.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>

#include <osg/MatrixTransform>

#include <string>
#include <vector>

class CVRSocket;

class TouchDesigner : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:        
        TouchDesigner();
        virtual ~TouchDesigner();
        
      	bool init();
        void preFrame();
        void menuCallback(cvr::MenuItem*);

    protected:
        cvr::SubMenu* _menu;
        cvr::MenuButton* _receiveButton;
        std::string _port;
        CVRSocket* _cvrsock;
        int _sockID; ///< socket descriptor

	void receiveGeometry();
};

#endif
