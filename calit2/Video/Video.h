#ifndef __VIDEO_H_
#define __VIDEO_H_
#include "videoplayerapi.h"
#include <osg/Geometry>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrKernel/CVRViewer.h>
#include <PluginMessageType.h>
#include <list>
#include <string>
#include <sstream>
#include <map>
#include <OpenThreads/Mutex>
#include "texturemanager.h"
//#include <cvrKernel/SceneObject.h>
//

struct PTSUpdate
{
	PTSUpdate() {}
	PTSUpdate(unsigned int gid, double pts) : gid(gid), pts(pts) {}
	unsigned int gid;
	double pts;
};

class Video : public cvr::CVRPlugin, cvr::MenuCallback, cvr::PerContextCallback
{
public:
	Video();
	virtual ~Video();

        void postFrame();
	void preFrame();
	bool init();
        void menuCallback(cvr::MenuItem * item);
	void perContextCallback(int contextid, cvr::PerContextCallback::PCCType type) const;
	void message(int type, char*& data, bool collaborative);
	
protected:
	int LoadVideoXML(const char* filename, std::list<std::string>& videoFilenames);
	void loadMenuItems(cvr::SubMenu* menu, const char* xmlFilename);
	void EncodePtsUpdates(const std::list<PTSUpdate>& updates, const size_t& buffSize, unsigned char* buffer) const;
	void DecodePtsUpdates(std::list<PTSUpdate>& updates, size_t buffSize, const unsigned char* buffer) const;
	std::list<std::string> ExplodeFilename(const char* filename);

	cvr::SubMenu* MLMenu;	
	cvr::SubMenu* loadMenu;	
	mutable cvr::SubMenu* removeMenu;	

	mutable VideoPlayerAPI m_videoplayer;
	mutable std::list<VideoMessageData> m_actionQueue;
	mutable std::string m_loadVideo;
	mutable std::list<cvr::MenuItem*> m_removeVideo;
	mutable std::map<unsigned int, TextureManager*> m_gidMap;

	mutable std::list<cvr::MenuItem*> m_menuDelete;
	mutable std::list<TextureManager*> m_managerAdd;
	mutable std::list<TextureManager*> m_managerLoad;
	mutable std::list<cvr::MenuItem*> m_menuAdd;

        mutable OpenThreads::Mutex m_updateMutex;
	mutable OpenThreads::Mutex m_initMutex;

	mutable std::list<PTSUpdate> m_ptsUpdateList;

	

	

};
	

bool videoNotifyFunction(VIDEOPLAYER_NOTIFICATION msg, unsigned int gid, void* obj, unsigned int param1, unsigned int param2, double param3);
#endif
