#ifndef VIS_MODE_H
#define VIS_MODE_H

class FlowPagedRenderer;

class VisMode
{
    public:
        virtual void initContext(int context)
        {
        }

        virtual void uinitContext(int context)
        {
        }

        virtual void frameStart(int context)
        {
        }

        virtual void preFrame()
        {
        }

        virtual void preDraw(int context)
        {
        }

        virtual void draw(int context)
        {
        }

        virtual void postFrame()
        {
        }

        FlowPagedRenderer * _renderer;
};

#endif
