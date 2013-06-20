#ifndef TEXTURE_MANAGER_H_
#define TEXTURE_MANAGER_H_

#include <GL/glew.h>
#include <osg/Geometry>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrKernel/CVRViewer.h>
#include <list>
#include <string>
#include <sstream>
#include <map>
#include <osg/Geometry>
#include <osg/Texture2D>
#include "rect.h"
#include <cvrKernel/SceneObject.h>
#include <vector>


class TextureObject : public osg::Texture2D::Texture2D::SubloadCallback
{
public:
	TextureObject(std::map<unsigned int, GLuint> tid) : _tid(tid)
	{
	}
	virtual ~TextureObject()
	{

	}
private:
	void load( const osg::Texture2D& texture, osg::State& state) const
	{
		unsigned int context = state.getContextID();
		glBindTexture(GL_TEXTURE_2D, _tid[context]);
		//printf("load %d\n", _tid[context]);
	}

	void subload( const osg::Texture2D& texture, osg::State& state) const
	{
		unsigned int context = state.getContextID();
		glBindTexture(GL_TEXTURE_2D, _tid[context]);
		//printf("subload %d for state %u\n", _tid[context], state.getContextID());
	}
	mutable std::map<unsigned int, GLuint> _tid; 

};

enum STEREO_EYE
{
	STEREO_LEFT,
	STEREO_RIGHT
};

class TextureManager
{
public:
	TextureManager(unsigned int gid);
	~TextureManager();
	void Draw();
	void DrawBorder();
	void MoveTo(double x, double y);
	void MoveBy(double dx, double dy);
	void Scale(double dx, double dy);
	bool IsAt(double x, double y);
	rect GetWindowBounds();
	unsigned int GetVideoGID();
	int GetVideoWidth();
	int GetVideoHeight();
	void load( const osg::Texture2D& texture, osg::State& state) const;
	void subload( const osg::Texture2D& texture, osg::State& state) const;
	void SetSceneObject(cvr::SceneObject* scene);
	cvr::SceneObject* GetSceneObject();

	void AddGID(unsigned int gid);
	void SetStereo(STEREO_EYE eye);
	bool IsStereo();
	STEREO_EYE GetStereo();

	osg::Geode* AddTexture(unsigned int gid, std::map<unsigned int, GLuint> texmap, unsigned int width, unsigned int height);
	unsigned int GetVideoCount() const;
	unsigned int GetVideoID(unsigned int idx) const;
	
	
private:
	cvr::SceneObject* _scene;

	std::vector<unsigned int> m_gidList;

	unsigned int m_gid;
	bool m_isStereo;
	STEREO_EYE m_eye;

};


#endif
