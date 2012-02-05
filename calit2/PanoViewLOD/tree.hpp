// Copyright (c) 2011 Robert Kooima
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#ifndef TREE_HPP
#define TREE_HPP

#include <cassert>
#include <cstdio>

//------------------------------------------------------------------------------

template <typename T> class tree
{
public:

    tree() : root(0), count(0) { }
   ~tree();
   
    T&   search(T, int);
    void insert(T, int);
    void remove(T);
    
    T    first();
    T    eject();
    void clear();
    void dump();
    int  size() const { return count; }

    void map(void (*)(T&, void *), void *);
    
    struct node
    {
        node *l;
        node *r;
        int   t;
        T     p;
    };
    
private:

    // Splay tree implementation.

    node *splay(node *t, T p);
    node *root;
    int  count;

    // LRU ejection handlers.

    node  *first(node *);
    node *oldest(node *);
    node *choose(node *);
    void replace(node *, node *, node *);
    void  unlink(node *, node *, node *);
    
    void map(node *, void (*)(T&, void *), void *);
};

//------------------------------------------------------------------------------

template <typename T> tree<T>::~tree()
{
    clear();
}

template <typename T> typename tree<T>::node *tree<T>::splay(node *t, T p)
{
    if (t)
    {
        node  n;
        node *l = &n;
        node *r = &n;
        node *y;

        n.l = 0;
        n.r = 0;

        for (;;)
        {
            if (p < t->p)
            {
                if (t->l != 0 && p < t->l->p)
                {
                    y    = t->l;
                    t->l = y->r;
                    y->r = t;
                    t    = y;
                }
                if (t->l == 0) break;

                r->l = t;
                r    = t;
                t    = t->l;
                continue;
            }

            if (t->p < p)
            {
                if (t->r != 0 && t->r->p < p)
                {
                    y    = t->r;
                    t->r = y->l;
                    y->l = t;
                    t    = y;
                }
                if (t->r == 0) break;

                l->r = t;
                l    = t;
                t    = t->r;
                continue;
            }
            break;
        }

        l->r = t->l;
        r->l = t->r;
        t->l = n.r;
        t->r = n.l;
    }
    return t;
}

//------------------------------------------------------------------------------
// The following functions implement the standard set of splay tree operations,
// search, insert, and remove. Each of these splays the requested node to the
// root, thus optimizing access to related items.

template <typename T> T& tree<T>::search(T p, int t)
{
    assert(root);
    root = splay(root, p);
    root->t = t;
    return root->p;
}

template <typename T> void tree<T>::insert(T p, int t)
{
    node *m = root = splay(root, p);

    if (m == 0 || p < m->p || m->p < p)
    {
        if (node *n = new node)
        {
            count++;
            n->p = p;
            n->t = t;
            root = n;

            if (m == 0)
            {
                n->l = 0;
                n->r = 0;
            }
            else if (p < m->p)
            {
                n->l = m->l;
                n->r = m;
                m->l = 0;
            }
            else if (m->p < p)
            {
                n->r = m->r;
                n->l = m;
                m->r = 0;
            }
        }
    }
}

template <typename T> void tree<T>::remove(T p)
{
    node *m = root = splay(root, p);

    if (m == 0 || p < m->p || m->p < p)
        return;
    else
    {
        if (m->l)
        {
            root    = splay(m->l, p);
            root->r = m->r;
        }
        else
            root    = m->r;

        delete m;
        count--;
    }
}

template <typename T> void tree<T>::clear()
{
    while (root) remove(root->p);
}

//------------------------------------------------------------------------------

template <typename T> T tree<T>::first()
{
    T p;
    
    assert(root);
    
    if (node *n = first(root))
    {
        p = n->p;
        unlink(root, 0, n);
        delete n;
        count--;
    }
    return p;
}

// Delete the node with the lowest time value. Do so in the fashion of a binary
// search tree, so as not to bias the splay.

template <typename T> T tree<T>::eject()
{
    T p;
    
    assert(root);

    if (node *n = oldest(root))
    {
        p = n->p;
        unlink(root, 0, n);
        delete n;
        count--;
    }
    return p;
}

//------------------------------------------------------------------------------

template <typename T> typename tree<T>::node *tree<T>::first(node *t)
{
    node *n = t;
    
    while (n && n->l)
           n  = n->l;
           
    return n;
}

// Seek the node with the lowest time value.

template <typename T> typename tree<T>::node *tree<T>::oldest(node *t)
{
    if (t)
    {
        node *l = oldest(t->l);
        node *r = oldest(t->r);
        
        if (l && r) return (l->t < r->t) ? l : r;
        if (l)      return l;
        if (r)      return r;
    }
    return t;
}

// Replace node a with node b in the children of node n. Beware the root.

template <typename T> void tree<T>::replace(node *n, node *a, node *b)
{
    if (n)
    {
        if (n->l == a)
            n->l = b;
        else
            n->r = b;
    }
    else root = b;
}

// Seek either the successor or predecessor of the given node and unlink it.

template <typename T> typename tree<T>::node *tree<T>::choose(node *n)
{
    node *p = n;
    node *c;
    
    if (lrand48() & 1)
    {
        for (c = n->l; c->r; c = c->r)
             p = c;

        p->r = 0;
        return c;
    }
    else
    {
        for (c = n->r; c->l; c = c->l)
             p = c;

        p->l = 0;
        return c;
    }
}

// Unlink node n from the tree rooted at node t. Node p tracks the parent.

template <typename T> void tree<T>::unlink(node *t, node *p, node *n)
{
    if (t)
    {
        if      (n->p < t->p) unlink(t->l, t, n);
        else if (t->p < n->p) unlink(t->r, t, n);
        else
        {
            if      (n->l == 0 && n->r == 0) replace(p, n, 0);
            else if (n->l == 0)              replace(p, n, n->r);
            else if (n->r == 0)              replace(p, n, n->l);
            else
            {
                node *c = choose(n);
                c->l = n->l;
                c->r = n->r;
                replace(p, n, c);
            }
        }
    }
}

//------------------------------------------------------------------------------

template <typename T> void tree<T>::map(node *t, void (*f)(T&, void *), void *p)
{
    if (t->l) map(t->l, f, p);
    f(t->p, p);
    if (t->r) map(t->r, f, p);
}

template <typename T> void tree<T>::map(void (*f)(T&, void *), void *p)
{
    if (root) map(root, f, p);
}

//------------------------------------------------------------------------------

#endif
