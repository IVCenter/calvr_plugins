#ifndef _CIRCULAR_STACK_
#define _CIRCULAR_STACK_

class CircularStack
{
    public:
        CircularStack(int size);
        ~CircularStack();

        void push(int size, float time);
        float getTimePerByte();

    protected:
        int _size;
        int * _sizeList;
        float * _timeList;
        int _index;   
};

#endif
