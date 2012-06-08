#include "TrackerTree.h"


using namespace std;

//constructer
TrackerTree::TrackerTree()
{
  /*
  TrackerNode* rootnode;
  rootnode->comment = "";
  rootnode->posInGeode = -1;
  
  rootnode->level = 1;
  
  rootnode->leftChild = NULL;
  rootnode->rightChild = NULL;
  */
  root = NULL;

}

//constructer that takes in a root node
TrackerTree::TrackerTree(TrackerNode* _root)
{
  root = _root;
}

TrackerTree::~TrackerTree()
{
  delete root;
}

//balance operation; right rotation
//input: parent node of the subtree to be skewed
//output: new parent node with balanced subtree
TrackerNode* TrackerTree::skew(TrackerNode* node)
{

  //check if balancing is needed
  if(node == NULL) return NULL;       
  else if(node->leftChild == NULL) return node;
  else if(node->leftChild-> level == node->level) {   //left horizontal link, needs balancing
    //swap pointers of the horizontal left links
    cerr<<"skewing"<<endl;
    TrackerNode* newNode = node->leftChild;           
    node->leftChild = newNode->rightChild;
    newNode->rightChild = node;
    return newNode;
  }
  else return node;
}

//balance operation; left rotation
//input: parent node of the subtree to be split
//output: new parent node with balanced subtree 
TrackerNode* TrackerTree::split(TrackerNode* node)
{

  //check if balancing is needed
  if(node == NULL) return NULL;
  else if(node->rightChild == NULL || node->rightChild->rightChild == NULL) return node;
  else if(node->level == node->rightChild->rightChild->level) { //two horizontal links, needs balancing
    //elevate the middle node and return it
    cerr<<"splitting"<<endl;
    TrackerNode* newNode = node->rightChild;
    node->rightChild = newNode->leftChild;
    newNode->leftChild = node;
    newNode->level++;
    return newNode;
  }
  else return node;
  
}

//inserts recursively in alphabetical order, skews and splits each stack level as necessary=================================================
//input: node to be inserted, root of tree
//output: root to a balanced tree with new inserted node
TrackerNode* TrackerTree::insert(TrackerNode* _node, TrackerNode* _root)
{
  cerr<<"inserting "<<_node->comment<<endl;
  //recursive insertion 
  if(_root == NULL) {
    _node->level = 1;
    return _node;
  }
  else if(_node->comment.compare(_root->comment) < 0) _root->leftChild = insert(_node, _root->leftChild);
  else if(_node->comment.compare(_root->comment) >= 0) _root->rightChild = insert(_node, _root->rightChild);
  
  //skew and split as necessary
  _root = skew(_root);
  _root = split(_root);
  
  return _root;
}
//insert, by string
TrackerNode* TrackerTree::insert(string _comment, TrackerNode* _root)
{
  cerr<<"inserting "<<_comment<<endl;

   TrackerNode* _node = new TrackerNode(_comment,-1);
  //recursive insertion 
  if(_root == NULL) {
    return _node;
  }
  else if(_node->comment.compare(_root->comment) < 0) _root->leftChild = insert(_node, _root->leftChild);
  else if(_node->comment.compare(_root->comment) >= 0) _root->rightChild = insert(_node, _root->rightChild);
  
  //skew and split as necessary
  _root = skew(_root);
  _root = split(_root);
  
  return _root;
}
//insert, by string and position in root
TrackerNode* TrackerTree::insert(string _comment, int _pos, TrackerNode* _root)
{
  cerr<<"inserting "<<_comment<<endl;

   TrackerNode* _node = new TrackerNode(_comment,_pos);
  //recursive insertion 
  if(_root == NULL) {
    return _node;
  }
  else if(_node->comment.compare(_root->comment) < 0) _root->leftChild = insert(_node, _root->leftChild);
  else if(_node->comment.compare(_root->comment) >= 0) _root->rightChild = insert(_node, _root->rightChild);
  
  //skew and split as necessary
  _root = skew(_root);
  _root = split(_root);
  
  return _root;
}
//insert, by string and position in root and pointer to Geode
TrackerNode* TrackerTree::insert(string _comment, int _pos, osg::Geode* _geo, TrackerNode* _root)
{
  cerr<<"inserting "<<_comment<<endl;

   TrackerNode* _node = new TrackerNode(_comment,_pos, _geo);
  //recursive insertion 
  if(_root == NULL) {
    return _node;
  }
  else if(_node->comment.compare(_root->comment) < 0) _root->leftChild = insert(_node, _root->leftChild);
  else if(_node->comment.compare(_root->comment) >= 0) _root->rightChild = insert(_node, _root->rightChild);
  
  //skew and split as necessary
  _root = skew(_root);
  _root = split(_root);
  
  return _root;
}
//================================================================================================================================


//these functions are used by remove
TrackerNode* getSuccessor(TrackerNode* node)
{
  cerr<<"retrieving successor"<<endl;
  if(node->leftChild == NULL) return node;
  else return getSuccessor(node->leftChild);
}
TrackerNode* getPredecessor(TrackerNode* node)
{
  cerr<<"retrieving predecessor"<<endl;
  if(node->rightChild == NULL) return node;
  else return getPredecessor(node->rightChild);
}
//used by remove
//input: root node to subtree to remove links that skip levels
//output: new root node with level decreased
TrackerNode* decreaseLevel(TrackerNode* _root)
{
  cerr<<"decreasing levels..."<<endl;
  int trueLevel; 
  if(_root->leftChild == NULL && _root->rightChild ==NULL) trueLevel = 1;
  else if(_root->rightChild == NULL) trueLevel = _root->leftChild->level + 1;
  else if(_root->leftChild == NULL) trueLevel = _root->rightChild->level;
  else trueLevel = min(_root->leftChild->level, _root->rightChild->level) + 1;
  if(trueLevel < _root->level) {
    _root->level = trueLevel;
    if(_root->rightChild != NULL && trueLevel < _root->rightChild->level) _root->rightChild->level = trueLevel;
  }
  return _root;
}

//removes a node by comment, first reducing the node to be removed to leaf level, then removing it
//rebalancing: decrease the level, if appropriate
//             skew & split
//input: comment to be removed, root of tree
//output: root to a balanced tree with appropriate node removed
TrackerNode* TrackerTree::remove(string _comment, TrackerNode* _root)
{
  if(_root != NULL) cerr<<"removing...\t current root: "<<_root->comment<<endl;
  else cerr<<"removing...\t current root: null"<<endl;

  //recursive remove
  if(_root == NULL) return _root;
  else if(_comment.compare(_root->comment) > 0) _root->rightChild = remove(_comment, _root->rightChild);
  else if(_comment.compare(_root->comment) < 0) _root->leftChild = remove(_comment, _root->leftChild);
  else {
    //if leaf node, simply remove // else reduce to leaf node using successors / predecessors
    if(_root->leftChild == NULL && _root->rightChild == NULL) return NULL;
    else if(_root->leftChild == NULL) {
      TrackerNode* successor = getSuccessor(_root->rightChild);
      cerr<<"successor retrieved: "<<successor->comment<<endl;
      _root->rightChild = remove(successor->comment,_root->rightChild);
      _root->comment = successor->comment;
    }
    else {
      TrackerNode* predecessor = getPredecessor(_root->leftChild);
      cerr<<"predecessor retrieved: "<<predecessor->comment<<endl;
      _root->leftChild = remove(predecessor->comment,_root->leftChild);
      _root->comment = predecessor->comment;
    }
  }
  
  //balancing: decrease the level of all nodes in this level if necessary
  //skew and split nodes in new level
  _root = decreaseLevel(_root);
  _root = skew(_root);
  if(_root->rightChild != NULL) _root->rightChild = skew(_root->rightChild);
  if(_root->rightChild != NULL && _root->rightChild->rightChild != NULL) _root->rightChild->rightChild = skew(_root->rightChild->rightChild);
  _root = split(_root);
  if(_root->rightChild != NULL) _root->rightChild = split(_root->rightChild);
  
  cerr<<"returning"<<endl;
  
  return _root;
  
}

//returns the node with the specified comment
//input: comment & root node to tree
//null if not found
TrackerNode* TrackerTree::get(string _comment, TrackerNode* _root)
{
  if(_root!=NULL) cerr<<"getting.... / current root = "<<_root->comment<<endl;
  if(_comment.compare(_root->comment) == 0) return _root;
  else if(_comment.compare(_root->comment) < 0 && _root->leftChild != NULL) return get(_comment, _root->leftChild);
  else if(_comment.compare(_root->comment) > 0 && _root->rightChild != NULL) return get(_comment, _root->rightChild);
  else return NULL; 
}

//prints out the entire tree
//for debug
//takes in root node of tree/subtree you want printed
void TrackerTree::printTree(TrackerNode* _root) 
{
  cerr<<"\nThe tree currently contains:\t"<<endl;
  vector<TrackerNode*> nodes;
  nodes.push_back(_root);
  
  while(!nodes.empty()) {
    cerr << " [" << nodes[0]->comment << " , level: " << nodes[0]->level << "]\t";
    if(nodes[0]->leftChild != NULL) nodes.push_back(nodes[0]->leftChild);
    if(nodes[0]->rightChild != NULL) nodes.push_back(nodes[0]->rightChild);
    nodes.erase(nodes.begin());
  }
}






