#ifndef LERPABLE_H
#define LERPABLE_H
    
    //a lerpable element:
    //      -has a floating-point value representing the lerp'ed amount
    //      -has an update() function that must be called that moves the value toward
    //          its goal
    //      -has a setValue() function, which resets the lerp and sets a new value
    //      -has a setValueImmediate() function, which overrides the lerpability of the object.


//T must have a copy constructor or something!
template<class T>
class Lerpable 
{
public:

    Lerpable(T startValue, T endValue)
    {
        lastValue = T(startValue);
        targetValue = T(endValue);
        percentageToTarget = 0.0f;
        speed = 1.0f;
        changed = true;
    }


    Lerpable(T startValue)
    {
        lastValue = T(startValue);
        targetValue = T(startValue);
        percentageToTarget = 1.0f;
        speed = 1.0f;
        changed = true;
    }
    
    Lerpable()
    {
        lastValue = T();
        targetValue = T();
        percentageToTarget = 1.0f;
        speed = 1.0f;
        changed = true;
    }

    ~Lerpable()
    {
        //nothing on the heap so nothing to deallocate
    }


    //this function should be called once per preframe
    void moveToTarget()
    {
        //perhaps allow the user to control the speed of the lerpable object?
        //the function even, maybe?
        
        if(percentageToTarget < 1.0f)
        {
            percentageToTarget += 0.01f * speed;
        
            //do a fancy lerp that moves faster in the middle and slower at the start and end!
            if(percentageToTarget <= 0.5f)
                percentageToTarget += 0.1f * percentageToTarget * speed;
            else
                percentageToTarget += 0.1f * (1.0f-percentageToTarget) * speed;
           
            if(percentageToTarget >= 1.0f)
            {
                //so that the next time we set a new target value
                //we start from this point. This isn't really needed,
                //since a call to setValue() should set lastValue to getValue()
                //(which should in turn be targetValue because percent >= 1.0f)
                //but it's just to be safe.
                lastValue = targetValue;
            } 
            changed = true;
        }
    }

    void setValue(T t)
    {
        //lerp from the current point to the target point now
        lastValue = getValue();
        targetValue = T(t);
                
        percentageToTarget = 0.0f;
        changed = true;
    }

    
    void setImmediateValue(T t)
    {
        //immediately jump to the target point
        lastValue = t;
        targetValue = t;

        percentageToTarget = 1.0f;
        changed = true;
    }

    T getValue()
    {
        //return (1.0f - percentageToTarget) * lastValue + percentageToTarget * targetValue;
        return lastValue * (1.0f - percentageToTarget) + targetValue * percentageToTarget;
    }
    
    T getTargetValue()
    {
        return targetValue;
    }

    float getPercentageToTarget()
    {
        return percentageToTarget;
    }
    
    //man! maybe this function shouldn't exist but until there's a classier 
    //way to do a smooth slowdown from an assumed constant speed, scratch it.
    void setPercentageToTarget(float f)
    {
        if(f > 1.0f || f < 0.0f)
            return;
        percentageToTarget = f;
    }

    float getSpeed() { return speed; }
    void setSpeed(float f) { speed=f;}

    bool hasChanged() { if(changed) { changed = 0; return true; } else return false; }
    
private:
    float percentageToTarget;

    float speed;
    bool changed;
    
    T lastValue;
    T targetValue;
};

#endif
