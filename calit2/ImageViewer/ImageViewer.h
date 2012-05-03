#ifndef CVR_IMAGE_VIEWER_PLUGIN_H
#define CVR_IMAGE_VIEWER_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <vector>
#include <string>

#include "ImageObject.h"

struct ImageInfo
{
    std::string name;
    bool stereo;
    std::string fileLeft;
    std::string fileRight;
    float aspectRatio;
    float width;
    float scale;
    osg::Vec3 position;
    osg::Quat rotation;
};

class ImageViewer : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        ImageViewer();
        virtual ~ImageViewer();

        bool init();
        void menuCallback(cvr::MenuItem * item);

    protected:
        void createLoadMenu(std::string tagBase, std::string tag, cvr::SubMenu * menu);

        std::string findFile(std::string name);

        cvr::SubMenu * _imageViewerMenu;
        cvr::SubMenu * _filesMenu;
        cvr::MenuButton * _removeButton;
        std::vector<cvr::MenuButton*> _fileButtons;
        std::vector<ImageInfo*> _files;

        std::vector<ImageObject*> _loadedImages;
        std::vector<cvr::MenuButton*> _deleteButtons;

        std::vector<std::string> _pathList;
};

#endif
