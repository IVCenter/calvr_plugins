#ifndef _TouchDesigner_
#define _TouchDesigner_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

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
