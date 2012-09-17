#ifndef _OSGPDF_
#define _OSGPDF_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/FileHandler.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osgWidget/PdfReader>

#include <osg/MatrixTransform>
#include <osg/ImageStream>
#include <osg/Uniform>

#include <string>
#include <vector>

#include <PdfReader.h>

class OsgPdf : public cvr::CVRPlugin, public cvr::MenuCallback ,public cvr::FileLoadCallback
{
    public:        
        OsgPdf();
        virtual ~OsgPdf();
	bool init();
        virtual bool loadFile(std::string file)
        {
            return loadFile(file,0);
        }
        virtual bool loadFile(std::string file, float width, bool tiledSO = false);
	void menuCallback(cvr::MenuItem * item);

        virtual void message(int type, char *&data, bool collaborative=false);
    protected:

	// container to hold pdf data
	struct PdfObject
        {
            std::string name;
	    cvr::SceneObject * scene;
	    PdfImage * pdf;
        };

	osg::Geometry * myCreateTexturedQuadGeometry(osg::Vec3, float width, float height, osg::Image* image);

        std::map<struct PdfObject*,cvr::MenuButton*> _previousMap;
        std::map<struct PdfObject*,cvr::MenuButton*> _nextMap;
        std::map<struct PdfObject*,cvr::MenuRangeValue*> _sliderMap;
        std::map<struct PdfObject*,cvr::MenuButton*> _deleteMap;
        std::vector<struct PdfObject*> _loadedPdfs;
};

#endif
