#ifndef _TouchDesigner_
#define _TouchDesigner_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <osg/MatrixTransform>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

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
        struct sockaddr_in _serverAddr;
        struct sockaddr_in _clientAddr;
        cvr::SubMenu* _menu;
        cvr::MenuButton* _receiveButton;
        std::string _port;
        int _sockID; ///< socket descriptor
        socklen_t _addrLen;
        
      	void receiveGeometry();
      	void initSocket();
      	void readSocket();
};

#endif
