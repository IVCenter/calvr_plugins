#include "Video.h"
#include <GL/glew.h>
#include "XMLConfig.h"
#include "stringreplace.h"
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/MenuButton.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/PluginHelper.h>

CVRPLUGIN(Video)

Video::Video() : m_removeVideo(0)
{

}

Video::~Video()
{
	cvr::CVRViewer::instance()->addPerContextPreDrawCallback(0);
	cvr::CVRViewer::instance()->addPerContextFrameStartCallback(0);
	
}

bool Video::init()
{
	std::cerr << "Video init" << std::endl;
	MLMenu = new cvr::SubMenu("Video", "Video");
//	MLMenu->setCallback(this);

	loadMenu = new cvr::SubMenu("Load Video", "Load");
	loadMenu->setCallback(this);

	removeMenu = new cvr::SubMenu("Delete Video", "Remove");
	removeMenu->setCallback(this);
	bool found = false;
	std::string fname = cvr::ConfigManager::getEntry("filename", "Plugin.Video.ConfigXML", "banana", &found);
	std::cout << fname << std::endl;
	loadMenuItems(loadMenu, fname.c_str());
	MLMenu->addItem(loadMenu);
	MLMenu->addItem(removeMenu);
	cvr::MenuSystem::instance()->addMenuItem(MLMenu);
	cvr::CVRViewer::instance()->addPerContextPreDrawCallback(this);
	//cvr::CVRViewer::instance()->addPerContextPostFinishCallback(this);
	//cvr::CVRViewer::instance()->addPerContextFrameStartCallback(this);
	return true;
}

void Video::loadMenuItems(cvr::SubMenu* menu, const char* xmlFilename)
{
	std::list<std::string> names;
	LoadVideoXML(xmlFilename, names);

	while (names.size())
	{
		cvr::MenuItem* mi = new cvr::MenuButton(names.front().c_str());
		mi->setCallback(this);
		menu->addItem(mi);
		names.pop_front();
	}
		
}

void Video::EncodePtsUpdates(const std::list<PTSUpdate>& updates, const size_t& buffSize, unsigned char* buffer) const
{
	size_t udsize = sizeof(unsigned int) * sizeof(double);
	int i = 0;
	for (std::list<PTSUpdate>::const_iterator iter = updates.begin(); iter != updates.end(); ++iter)
	{
		const PTSUpdate& update = *iter;
		memcpy(&buffer[i*udsize], &update.gid, sizeof(update.gid));
		memcpy(&buffer[i*udsize + sizeof(update.gid)], &update.pts, sizeof(update.pts));
		i++;
	}
}

void Video::DecodePtsUpdates(std::list<PTSUpdate>& updates, size_t buffSize, const unsigned char* buffer) const
{
	size_t udsize = sizeof(unsigned int) * sizeof(double);
	const unsigned char* b = buffer;
	while (b <= buffer + buffSize - udsize)
	{
		PTSUpdate update;
		memcpy(&update.gid, b, sizeof(update.gid));
		b+= sizeof(update.gid);
		memcpy(&update.pts, b, sizeof(update.pts));
		b+= sizeof(update.pts);
		updates.push_back(update);
	}
}

void Video::postFrame()
{
	std::list<PTSUpdate> ptsList;
	unsigned char* ptsData = 0;
	// alloc data for pts
	bool isHead = cvr::ComController::instance()->isMaster();
	unsigned int i = 0;
	//isHead = true;
	if (isHead)
	{
		for (std::map<unsigned int, TextureManager*>::iterator iter = m_gidMap.begin(); iter != m_gidMap.end(); ++iter)
		{
			TextureManager* manager = iter->second;
			unsigned int gid = manager->GetVideoID(0);
			//printf("CheckUpdateVideo for video %x\n", gid);
			// head node updates the video
			// XXX split UpdateVideo into a call to just check if a new frame needs to be loaded, but not load the frame yet, just get the frame PTS, and then another call to actually load the texture (ideally preloaded)
			bool videoUpdate = m_videoplayer.CheckUpdateVideo(gid, true);

			if (videoUpdate)
			{
				//printf("CheckUpdateVideo returned true\n");
				PTSUpdate update(gid, m_videoplayer.GetVideoPts(gid));
				ptsList.push_back(update);
			}
		}


		size_t udsize = sizeof(unsigned int) * sizeof(double);
		unsigned int buffSize = ptsList.size() * udsize;
		unsigned char* buffer = new unsigned char[buffSize + sizeof(unsigned int)];
		memcpy(buffer, &buffSize, sizeof(unsigned int));
		EncodePtsUpdates(ptsList, buffSize, &buffer[sizeof(unsigned int)]);
		cvr::ComController::instance()->sendSlaves(buffer, buffSize + sizeof(unsigned int));
	}
	else
	{
		unsigned int buffSize;
		cvr::ComController::instance()->readMaster((unsigned char*)&buffSize, sizeof(unsigned int));
		unsigned char* buffer = new unsigned char[buffSize];
		cvr::ComController::instance()->readMaster(buffer, buffSize);

		DecodePtsUpdates(ptsList, buffSize, buffer);
	}


	m_ptsUpdateList.clear();
	m_ptsUpdateList.splice(m_ptsUpdateList.begin(), ptsList);

}

void Video::preFrame()
{
	while (m_sceneDelete.size())
	{
		m_sceneDelete.front()->detachFromScene();;
		delete m_sceneDelete.front();
		m_sceneDelete.pop_front();
	}
	while (m_menuDelete.size())
	{
		delete m_menuDelete.front();
		m_menuDelete.pop_front();
	}
	while (m_sceneAdd.size())
	{
		m_sceneAdd.front()->attachToScene();
		m_sceneAdd.pop_front();
	}
	while (m_menuAdd.size())
	{
		removeMenu->addItem(m_menuAdd.front());
		m_menuAdd.pop_front();
	}
}

void Video::menuCallback(cvr::MenuItem* item)
{
	if (item->getType() == cvr::BUTTON)
	{
		if (item->getParent() == loadMenu)
		{
			m_loadVideo = static_cast<cvr::MenuButton*>(item)->getText();
			std::cout << "Load video file " << m_loadVideo.c_str() << std::endl;
		}
		else if (item->getParent() == removeMenu)
		{
			m_removeVideo = item;
			removeMenu->removeItem(item);
			//delete manager;
			//delete item;
		}
		else
		{
			std::cout << "Unknown menu item parent." << std::endl;
		}
	}
	if (item == loadMenu)
	{
	}
}


void Video::perContextCallback(int contextid, cvr::PerContextCallback::PCCType type) const
{
	static int init = 0;
	stopwatch cbt;
	cbt.start();
	if (init == 0)
	{
		m_initMutex.lock();
		if (init == 0)
		{
			init = 1;
			glewInit();
			m_videoplayer.RegisterNotificationFunction(videoNotifyFunction, 0);
			bool isHead = cvr::ComController::instance()->isMaster();
			m_videoplayer.init(isHead, false);
			//m_videoplayer.init(true, true);
			//m_videoplayer.init(false, false);
			//gid = m_videoplayer.LoadVideoFile("/mnt/pointstar/cars_1080p.mov", true);
			//printf("Loaded cars\n");
		}
		m_initMutex.unlock();

	}
	if (contextid == 0)
	{

		if (m_loadVideo.size())
		{
			unsigned int gid = 0;
			std::cout << "Trying to load video " << m_loadVideo << std::endl;
			gid = m_videoplayer.LoadVideoFile(m_loadVideo.c_str(), true);

			TextureManager* manager = new TextureManager(gid);
			int nrows = 1;
			int ncols = 1;
			if (gid & 0x80000000) // multi-tile video
			{
				nrows = ((gid & 0x7E000000) >> 25) + 1;
				ncols = ((gid & 0x01F80000) >> 19) + 1;
			}

			std::cout << "Loading video gid " << gid << ", with " << nrows << " rows and " << ncols << " cols." << std::endl;
			cvr::SceneObject* scene = new cvr::SceneObject("Video Scene", true, true, false, false, true);

			for (int y = 0; y < nrows; y++)
			{
				for (int x = 0; x < ncols; x++)
				{
					unsigned int myid = gid | (y << 13) | (x << 7); 
					//XXX enable video on the head node.  This doesn't seem to work though
					//m_videoplayer.EnableHeadVideo(true, myid);
					int width = m_videoplayer.GetVideoWidth(myid);
					int height = m_videoplayer.GetVideoHeight(myid);
					GLuint tex = m_videoplayer.GetTextureID(myid);
					std::cout << "Texture id: " << tex << std::endl;
					osg::Geode* to = manager->AddTexture(myid, tex, width, height);
					printf("added manager with gid %d, width %d, height %d\n", gid, width, height);
					scene->addChild(to);
				}
			}


			m_gidMap[gid] = manager;


			cvr::PluginHelper::registerSceneObject(scene, "Video");
			manager->SetSceneObject(scene);
			m_sceneAdd.push_back(scene);

			cvr::MenuButton* button = new cvr::MenuButton(m_loadVideo);
			button->setExtraData(manager);
			button->setCallback(const_cast<Video*>(this));
			m_menuAdd.push_back(button);


			m_loadVideo.clear();
		}	
		if (m_removeVideo)
		{
			TextureManager* manager = static_cast<TextureManager*>(m_removeVideo->getExtraData());

			unsigned int gidcount = manager->GetVideoCount();
			for (unsigned int i = 0; i < gidcount; i++) // only update the first, it will update all
			{	
				unsigned int gid = manager->GetVideoID(i);

				std::cout << "Removing video file " << gid << std::endl;
				m_videoplayer.RemoveVideoFile(gid);
			}
			m_gidMap.erase(manager->GetVideoGID());
			m_sceneDelete.push_back(manager->GetSceneObject());
			delete manager;
			m_menuDelete.push_back(m_removeVideo);
			m_removeVideo = 0;
		}
	}


	stopwatch timer;
	double videoTime = 0;
	double textureTime = 0;
	for (std::list<PTSUpdate>::iterator iter = m_ptsUpdateList.begin(); iter != m_ptsUpdateList.end(); ++iter)
	{
		PTSUpdate update = *iter;
		TextureManager* manager = m_gidMap[update.gid];

		//printf("PTSUpdate for video group %x\n", update.gid);
		unsigned int gidcount = manager->GetVideoCount();
		for (unsigned int i = 0; i < gidcount; i++)
		{
			timer.start();
			unsigned int gid = manager->GetVideoID(i);
			m_videoplayer.SetCurrentDrawPts(gid, update.pts);
			//printf("Updating video %d with gid %x\n", i, gid);
			bool isupdate = m_videoplayer.UpdateVideo(gid, false);
			videoTime += timer.getTimeMS();
			timer.start();
			if (isupdate)
			{
				m_videoplayer.UpdateTexture(gid);
				textureTime += timer.getTimeMS();
				timer.start();
		//		printf("Updated video %x to pts %.4lf\n", gid, update.pts);
			}
			else
			{
		//		printf("No update for video %x to pts %.4lf\n", gid, update.pts);
			}
		}
	}
	//printf("Updated %d videos, times took %.4lfms for video and %.4lfms for texture\n", m_ptsUpdateList.size(), videoTime, textureTime);
	m_ptsUpdateList.clear();

	/*
	for (std::map<unsigned int, TextureManager*>::iterator iter = m_gidMap.begin(); iter != m_gidMap.end(); ++iter)
	{
		TextureManager* manager = iter->second;
		unsigned int gid = manager->GetVideoID(0);
		//printf("Updating video %x\n", gid);
		// head node updates the video
		bool isHead = cvr::ComController::instance()->isMaster();

		bool videoUpdate = m_videoplayer.UpdateVideo(gid, true);
		// get the video pts value, send to everyone, and set the draw pts
		double pts = m_videoplayer.GetVideoPts(gid);

		if (videoUpdate)
		{
			m_videoplayer.SetCurrentDrawPts(gid, pts);
			m_videoplayer.UpdateTexture(gid);
			//printf("Update video\n");
			unsigned int gidcount = manager->GetVideoCount();
			for (unsigned int i = 1; i < gidcount; i++) // only update the first, it will update all
			{	
				gid = manager->GetVideoID(i);

				m_videoplayer.SetCurrentDrawPts(gid, pts);
				if (videoUpdate)
				{
				if (m_videoplayer.UpdateVideo(gid, false) != videoUpdate)
				{
					printf("Video update doesn't agree\n");
				}
				}
				if (m_videoplayer.GetVideoPts(gid) != pts)
					printf("Non matching PTS values\n");
				
				// once draw pts is updated, update the texture
				if (videoUpdate)
					m_videoplayer.UpdateTexture(gid);
				//m_videoplayer.PreDraw();
			}
		}
		//else
		{
			//printf("No update\n");
		}
	}
	*/
	//printf("Full callback took %.4lfms\n", cbt.getTimeMS());
}
 
bool videoNotifyFunction(VIDEOPLAYER_NOTIFICATION msg, unsigned int gid, void* obj, unsigned int param1, unsigned int param2, double param3)
{
	if (msg == VIDEOPLAYER_OPEN)
	{
		printf("Video %d opened with size (%d, %d)\n", gid, param1, param2);

		int width = param1;
		int height = param2;
		
		/*
		TextureManager* manager = new TextureManager(gid, width, height);
		m_gidMap[gid] = manager;
		m_gidOrderList.push_back(manager);
		printf("added manager with gid %d\n", gid);
		m_videoMap[filecount-1].push_back(gid);
		manager->Scale(-(width - width * (scaleFactor))/2, 0);

		manager->MoveTo(m_usedWidth - width * (1-scaleFactor) / 2, m_usedHeight - height * (1-scaleFactor) / 2);

		m_usedWidth += width * scaleFactor;

		if (m_usedWidth - ((nx-1) * width * scaleFactor) > 1)
		{
			m_usedWidth = 0;
			m_usedHeight += height * scaleFactor;
		}
		*/
		
	}

	if (msg == VIDEOPLAYER_CLOSE)
	{

		int width = param1;
		int height = param2;

		/*
		m_gidOrderList.remove(m_gidMap[gid]);
		delete m_gidMap[gid];
		m_gidMap.erase(gid);
		

		m_usedWidth -= width * scaleFactor;

		if (m_usedWidth < -0.1)
		{
			m_usedWidth = ((nx - 1) * width * scaleFactor);
			m_usedHeight -= height * scaleFactor;
		}
		*/

	}
	return true;
}

std::list<std::string> Video::ExplodeFilename(const char* filename)
{
	std::list<std::string> names;
	int start = -1;
	int end = -1;
	for (int i = 0; filename[i] != 0; i++)
	{
		if (filename[i] == '[')
			start = i;
		if (filename[i] == ']')
		{
			end = i;
			break;
		}
	}
	if (start >= 0 && end > start)
	{
		
		char* glob = 0;
		int globlen = end - start - 1;
		if (globlen > 0)
		{
			glob = new char[globlen + 1];
			memcpy(glob, &filename[start+1], globlen);
			glob[globlen] = 0;
			int* numbers;
			int cNums = stringreplace::grabNumberList(glob, &numbers);
			
			std::stringstream tst;
			size_t digitLength = 0;
			if (cNums)
			{
				tst << numbers[cNums - 1];
				digitLength = tst.str().length();
			}

			for (int i = 0; i < cNums; i++)
			{
				
				
				std::string str;
				str.append(filename, start);

				std::stringstream rep;
				rep << numbers[i];
				if (rep.str().length() < digitLength)
				{
					str.append(digitLength - rep.str().length(), '0');
				}
				str.append(rep.str());				
				str.append(&filename[end+1]);
				names.push_back(str);
			}
			delete[] numbers;
		}
	}
	names.push_back(std::string(filename));
	return names;
}

int Video::LoadVideoXML(const char* filename, std::list<std::string>& videoFilenames)
{
	XMLConfig config(filename);

	if (config.isLoaded())
	{
		if (!config.enterSubSection("file"))
			return 0;
		const char* src;
		const char* name;
		src = config.getAttribute("src");
		name = config.getAttribute("name");
		if (src && name)
		{

			std::list<std::string> src_values = ExplodeFilename(src);
			while (src_values.size())
			{
				videoFilenames.push_back(src_values.front());
				src_values.pop_front();
			}
		}
		while(config.enterSiblingSection("file"))
		{
			src = config.getAttribute("src");
			name = config.getAttribute("name");
			if (src && name)
			{

				std::list<std::string> src_values = ExplodeFilename(src);
				while (src_values.size())
				{
					videoFilenames.push_back(src_values.front());
					src_values.pop_front();
				}
			}
		}
	}
	else
	{
		printf("Unable to load config file %s\n", filename);
		return -1;
	}

	return 0;

} 
