#include <osg/NodeVisitor>

class TransparencyVisitor : public osg::NodeVisitor
{
    public:
        TransparencyVisitor();
        ~TransparencyVisitor();

        enum Mode {
            ALL_OPAQUE,
            ALL_TRANSPARENT
        };

        void setMode(Mode mode);
	Mode getMode();

        virtual void apply(osg::Geode&);

    protected:
        Mode _currentMode;
};
