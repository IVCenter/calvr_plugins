#ifndef _TrackerTree_
#define _TrackerTree_

#include <string>
#include <vector>
#include <iostream>
#include <math.h>

#include<osg/Geode>

using namespace std;
using namespace osg;

//Node Struct
//Contains comment and position in geode
//pointers to parent, left child, and right child
struct TrackerNode {

    string comment;
    int posInRoot;                //child index under Root
    osg::Geode* ptrToGeode;       //pointer to the Geode

    TrackerNode* leftChild;
    TrackerNode* rightChild;

    int level;        //used to balance the tree
    
    //default constructor
    TrackerNode() {
      comment = "";
      posInRoot = -1;
      ptrToGeode = NULL;

      leftChild = NULL;
      rightChild = NULL;
      
      level = 1;
    }

    //constructor with param position in root
    TrackerNode(string _comment, int _pos) {
      comment = _comment;
      posInRoot = _pos;
      ptrToGeode = NULL;

      leftChild = NULL;
      rightChild = NULL;
      
      level = 1;
    }
    //constructor with param ptr to Geode
    TrackerNode(string _comment, osg::Geode* _ptr) {
      comment = _comment;
      posInRoot = -1;
      ptrToGeode = _ptr;

      leftChild = NULL;
      rightChild = NULL;
      
      level = 1;
    }
    //constructor with both params
    TrackerNode(string _comment, int _pos, osg::Geode* _ptr) {
      comment = _comment;
      posInRoot = _pos;
      ptrToGeode = _ptr;

      leftChild = NULL;
      rightChild = NULL;
      
      level = 1;
    }
    
    ~TrackerNode() {
      delete leftChild;
      delete rightChild;
      delete this;
    }

};


//Self Balancing AA Binary Tree
//   1. The level of a leaf node is one.
//   2. The level of a left child is exactly one less than that of its parent.
//   3. The level of a right child is equal to or one less than that of its parent.
//   4. The level of a right grandchild is strictly less than that of its grandparent.
//   5. Every node of level greater than one must have two children.
// * Equal levels are considered a "horizontal link"
// * Horizontal Links are only allowed to the right

class TrackerTree
{
public:
  
  TrackerTree();    //constructor

  ~TrackerTree();   //destructor
  
  TrackerTree(TrackerNode*);    //constructor that takes in a root node

  TrackerNode* insert(TrackerNode*, TrackerNode*);        //inserts a node, decided by alphabetical order
  TrackerNode* insert(string, TrackerNode*);              //insert by comment
  TrackerNode* insert(string,int,TrackerNode*);           //insert by comment and position
  TrackerNode* insert(string, int, osg::Geode*, TrackerNode*); //insert by comment with geode and position in root
  
  TrackerNode* remove(string, TrackerNode*);      //removes a node by comment
  
  TrackerNode* skew(TrackerNode*);    //balance operation, right rotation to resolve a left horizontal link
  TrackerNode* split(TrackerNode*);               //balance operation, left rotation to resolve two horizontal links
  
  TrackerNode* get(string, TrackerNode*);      //gets a child matching the comment

  void printTree(TrackerNode*);                 //prints out the entire tree

  TrackerNode* root;

};

#endif
