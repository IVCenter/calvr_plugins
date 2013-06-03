//  Copyright (C) 2005-2011 Robert Kooima
//
//  THUMB is free software; you can redistribute it and/or modify it under
//  the terms of  the GNU General Public License as  published by the Free
//  Software  Foundation;  either version 2  of the  License,  or (at your
//  option) any later version.
//
//  This program  is distributed in the  hope that it will  be useful, but
//  WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of
//  MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU
//  General Public License for more details.

#ifndef SPH_CACHE_HPP
#define SPH_CACHE_HPP

#include <vector>
#include <string>
#include <list>
#include <map>

#include <GL/glew.h>
#include <tiffio.h>
#include <SDL.h>
#include <SDL_thread.h>

#include "queue.hpp"

#include <list>

#include "DiskCache.h"

class sph_cache;
class PanoViewLOD;
//------------------------------------------------------------------------------

template <typename T> class fifo : public std::list <T>
{
public:

    void enq(T p) {
        this->push_back(p);
    }
    
    T deq() {
        T p = this->front();
        this->pop_front();
        return p;
    }
};

//------------------------------------------------------------------------------

struct sph_item
{
    sph_item()             : f(-1), i(-1) { }
    sph_item(int f, int i) : f( f), i( i) { }

    int f;
    int i;
    
    bool valid() const { return (f >= 0 && i >= 0); }
    
    bool operator<(const sph_item& that) const {
        if (i == that.i)
            return f < that.f;
        else
            return i < that.i;
    }
};

//------------------------------------------------------------------------------

struct sph_page : public sph_item
{
    sph_page()                              : sph_item(    ), o(0), t(0) { }
    sph_page(int f, int i)                  : sph_item(f, i), o(0), t(0) { }
    sph_page(int f, int i, GLuint o)        : sph_item(f, i), o(o), t(0) { }
    sph_page(int f, int i, GLuint o, int t) : sph_item(f, i), o(o), t(t) { }

    GLuint o;
    int    t;
};

//------------------------------------------------------------------------------

struct sph_task : public sph_item
{
    sph_task()             : sph_item(    ), u(0), p(0), cache(NULL), timestamp(0), valid(true) { }
    sph_task(int f, int i) : sph_item(i, f), u(0), p(0), cache(NULL), timestamp(0), valid(true) { }
    sph_task(int f, int i, GLuint u, GLsizei s, sph_cache * c, int t);
    
    GLuint u;
    void  *p;
    int timestamp;
    unsigned int size;
    
    bool make_texture(GLuint, uint32, uint32, uint16, uint16);
    void load_texture(TIFF *, uint32, uint32, uint16, uint16);
    void dump_texture();

    bool valid;
    sph_cache * cache;
};

//------------------------------------------------------------------------------

struct sph_file
{
    sph_file() {}
    sph_file(const std::string& name);
    
    std::string name;
    uint32 w, h;
    uint16 c, b;
};

//------------------------------------------------------------------------------

class sph_set
{
public:

    sph_set(int size) : size(size) { }
   ~sph_set();
   
    int  count()  const { return int(m.size()); }
    bool full()  const { return (count() >= size); }
    bool empty() const { return (m.empty()); }

    sph_page search(sph_page, int);
    void     insert(sph_page, int);
    void     remove(sph_page);

    sph_page  eject(int, int);
    void      draw();
    
private:

    std::map<sph_page, int> m;

    int size;
};

//------------------------------------------------------------------------------

class sph_cache
{
    friend class JobThread;
    friend class DiskCache;
    friend class PanoViewLOD;
    friend class PanoDrawableLOD;
public:

    sph_cache(int);
   ~sph_cache();

    int    add_file(const std::string&);
    GLuint get_page(int, int, int, int&);
    GLuint get_fill() { return filler; }
    
    void update(int);
    void flush();
    void draw();
    
    void set_debug(bool b) { debug = b; }

private:

    //std::vector<sph_file> files;
    std::map<int,sph_file> files;
    
    sph_set pages;
    sph_set waits;
    
    queue<sph_task> needs;
    queueNoCap<sph_task> loads;

    //std::list<sph_task> _delayList;

    fifo<GLuint> pbos;
        
    GLsizei pagelen(int);
    GLuint  filler;
    bool    debug;
    
    SDL_Thread *thread[4];
    
    friend int loader(void *);

    static DiskCache * _diskCache;
    float _maxTime;
    bool _useDiskCache;
};

//------------------------------------------------------------------------------

int up(TIFF *T, int i);
int dn(TIFF *T, int i);

#endif
