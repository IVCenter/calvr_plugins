/* osgCompute - Copyright (C) 2008-2009 SVT Group
*                                                                     
* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of
* the License, or (at your option) any later version.
*                                                                     
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of 
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesse General Public License for more details.
*
* The full license is in LICENSE file included with this distribution.
*/
#ifndef PTCLDEMO_PTCL_MOVER
#define PTCLDEMO_PTCL_MOVER 1

#include <osg/ref_ptr>
#include <osgCompute/Module>
#include <osgCompute/Memory>

namespace PtclDemo
{
    class AdvanceTime : public osgCompute::Resource
    {
    public:
        AdvanceTime() : osgCompute::Resource() {}

        META_Object(PtclDemo,AdvanceTime)

        osg::FrameStamp* _fs;

    protected:
        virtual ~AdvanceTime() {}

    private:
        AdvanceTime(const AdvanceTime&, const osg::CopyOp& ) {} 
        inline AdvanceTime &operator=(const AdvanceTime &) { return *this; }
    };

    class PtclMover : public osgCompute::Module 
    {
    public:
        PtclMover() : osgCompute::Module() {clearLocal();}

        META_Object( PtclDemo, PtclMover )

        // Modules have to implement at least this
        // three methods:
        virtual bool init();
        virtual void launch();
        virtual void acceptResource( osgCompute::Resource& resource );

        virtual void clear() { clearLocal(); osgCompute::Module::clear(); }
    protected:
        virtual ~PtclMover();
        void clearLocal();

        double                              _lastTime;
        bool						        _firstFrame;

        unsigned int                        _numBlocks;
        unsigned int                        _numThreads;

        osg::ref_ptr<AdvanceTime>           _timer;
        osg::ref_ptr<osgCompute::Memory>    _ptcls;

    private:
        PtclMover(const PtclMover&, const osg::CopyOp& ) {} 
        inline PtclMover &operator=(const PtclMover &) { return *this; }
    };

};

#endif // PTCLDEMO_PTCL_MOVER
