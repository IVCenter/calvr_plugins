#ifndef CVR_IMAGE_SEQUENCE_H
#define CVR_IMAGE_SEQUENCE_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>

#include <vector>
#include <string>

struct SequenceSet
{
    std::string path;
    int start;
    int frames;
};

class ImageSequence : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        ImageSequence();
        virtual ~ImageSequence();

        bool init();
        void preFrame();

        void menuCallback(cvr::MenuItem * item);

    protected:
        cvr::SubMenu * _isMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuButton * _removeButton;
        std::vector<cvr::MenuButton*> _loadButtons;
        std::vector<SequenceSet*> _sets;

        cvr::SceneObject * _activeObject;

        bool _autoStart;
        int _autoStartIndex;
};

#endif
