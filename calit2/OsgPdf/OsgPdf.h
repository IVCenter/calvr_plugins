#ifndef _OSGPDF_
#define _OSGPDF_

#include <kernel/CVRPlugin.h>
#include <kernel/FileHandler.h>
#include <kernel/SceneObject.h>
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuCheckbox.h>

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
        virtual bool loadFile(std::string file);
	void menuCallback(cvr::MenuItem * item);

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
