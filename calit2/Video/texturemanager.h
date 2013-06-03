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
	TextureObject(GLuint tid) : _tid(tid)
	{
	}
	virtual ~TextureObject()
	{

	}
	void UpdateTexture(GLuint tid)
	{
		_tid = tid;
	}
private:
	void load( const osg::Texture2D& texture, osg::State& state) const
	{
		glBindTexture(GL_TEXTURE_2D, _tid);
	}

	void subload( const osg::Texture2D& texture, osg::State& state) const
	{
		glBindTexture(GL_TEXTURE_2D, _tid);
	}

	GLuint _tid;

};

class TextureManager
{
public:
	TextureManager(unsigned int gid);
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

	osg::Geode* AddTexture(unsigned int gid, GLuint tid, unsigned int width, unsigned int height);
	unsigned int GetVideoCount() const;
	unsigned int GetVideoID(unsigned int idx) const;
	
	
private:
	cvr::SceneObject* _scene;

	std::vector<unsigned int> m_gidList;
	std::map<unsigned int, GLuint> m_texidMap;

	unsigned int m_gid;

};


#endif
