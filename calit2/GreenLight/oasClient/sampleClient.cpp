/* Client code in C - Converted to C++ */

#include "OASSound.h"
#include <cmath>
#include <sys/time.h>

using namespace std;

void wait(double duration);
void test1();
void test2();
void test3();
void test4();
void test0();

#define RESOLUTION 5
#define PI 3.14159265

oasclient::OASSound *heli, *boing, *turbine, *wave;

int main(void)
{
    if (!oasclient::OASClientInterface::initialize("137.110.119.175", 31232))
    {
        std::cerr << "Could not set up connection to server!\n";
        exit(1);
    }

    // Set up heli sound
    heli = new oasclient::OASSound("/home/schowkwa/cache", "helikopter.wav");
    if (!heli->isValid())
    {
        std::cerr << "Could not create helicopter sound!\n";
        exit(1);
    }

    turbine = new oasclient::OASSound("home/schowkwa/cache", "turbine.wav");

    if (!turbine->isValid())
    {
        std::cerr << "Could not create turbine sound!\n";
        exit(1);
    }

   // turbine->setLoop(true);
    heli->setLoop(true);
    heli->play();
    oasclient::OASClientInterface::setListenerGain(1.0);
//    turbine->play();
    wait(2);
    heli->stop();
    
	// BEGIN TEST 4 //
    // Doppler effect. Place sound forwards and to the left. Have sound move to the right, with a specified velocity.

    test4();

    oasclient::OASClientInterface::shutdown();

    return 0;
}

////////////
// TEST 0 //
////////////
void test0()
{
    // Play stationary sound

    //heli->setLoop(true);
    heli->setGain(0.5);
    heli->play();
    wait(7);
    heli->stop();
}

////////////
// TEST 1 //
////////////
void test1()
{
	float x, y, z, r, theta;
	x = y = z = r = theta = 0;
	
    // Loop in the x-y plane, starting from center, going to the right, and then counter-clockwise 360 degrees

	turbine->setLoop(true);
	turbine->play();
    wait(2);

    for (r = 0; r < 10; r += 0.25 / RESOLUTION)
    {
        turbine->setPosition(r, 0, 0);

        wait(0.1 / RESOLUTION);
    }

    for (theta = 0; theta < 360; theta += 2.5 / RESOLUTION)
    {
        x = r * cos(theta * PI / 180);
        y = r * sin(theta * PI / 180);
        turbine->setPosition(x, y, 0);
        wait(0.1 / RESOLUTION);
    }

    turbine->stop();
    wait(2);
}

////////////
// TEST 2 //
////////////
void test2()
{
	float x, y, z, r, theta;
	x = y = z = r = theta = 0;
	
    // Loop in the y-z plane, starting from directly in front, going overhead, then directly behind, then below, and back to the front.

    // STATUS: FAIL - vertical distinction of sound is difficult

    x = z = 0;
    y = 10;
    r = 10;
    heli->setLoop(true);
    heli->setPosition(0, 10, 0);
    heli->play();

    for (theta = 0; theta < 360; theta += 2.5 / RESOLUTION)
    {
        y = r * cos(theta * PI / 180);
        z = r * sin(theta * PI / 180);
        heli->setPosition(0, y, z);
        wait(0.1 / RESOLUTION);
    }

    heli->stop();
    wait(2);
}

////////////
// TEST 3 //
////////////
void test3()
{
	float x, y, z, r, theta;
	x = y = z = r = theta = 0;
	
    // BEGIN TEST 3 //
    // Place sound in center, move directly forwards and then backwards, staying on the y axis

    // STATUS: FAIL - gain attenuation works, but the distinction between forwards and backwards is small

	heli->setPosition(0, 0, 0);
	heli->setLoop(true);
	heli->play();

    for (y = 0; y < 10; y += 0.25 / RESOLUTION)
    {
        heli->setPosition(0, y, 0);
        wait(0.1 / RESOLUTION);
    }

    for (; y >= -10; y-= 0.25 / RESOLUTION)
    {;
        heli->setPosition(0, y, 0);
        wait(0.1 / RESOLUTION);
    }

    heli->stop();
    wait(2);
}

////////////
// TEST 4 //
////////////
void test4()
{
	float x, y, z, r, theta, velocityX, gain;
	x = y = z = r = theta = 0;
	
	// BEGIN TEST 4 //
    // Doppler effect. Place sound forwards and to the left. Have sound move to the right, with a specified velocity.

	velocityX = 30;
	heli->setPosition(-10, 3, 0);
	heli->setVelocity(velocityX, 0, 0);
	heli->setLoop(true);
	heli->play();

    for (x = -10; x < 10; x += 0.25 / RESOLUTION)
    {
        heli->setPosition(x, 3, 0);
        wait(0.1 / 3);
    }
    
    gain = 1.0;

    for (; x < 17.5; x += 0.25 / RESOLUTION)
    {
        velocityX -= 1.0 / RESOLUTION;
        heli->setVelocity(velocityX, 0, 0);
        heli->setPosition(x, 3, 0);
        gain -= 0.02 / RESOLUTION;
        heli->setGain(gain);
        wait (0.1 / 3);
    }

    for (; x > 10; x -= 0.25 / RESOLUTION)
    {
        velocityX -= 1.0 / RESOLUTION;
        heli->setVelocity(velocityX, 0, 0);
        heli->setPosition(x, 3, 0);
        gain += 0.02 / RESOLUTION;
        heli->setGain(gain);
        wait (0.1 / 3);
    }


//    heli->setVelocity(-30, 0, 0);
    for (; x > -10; x -= 0.25 / RESOLUTION)
    {
        heli->setPosition(x, 3, 0);
        wait(0.1 / 3);
    }
    heli->stop();
    wait(2);
}

void wait(double duration)
{
    struct timeval start, end;
    gettimeofday(&start, NULL);
    double diff;

    do
    {
        gettimeofday(&end, NULL);
        diff = ((end.tv_sec + ((double) end.tv_usec / 1000000.0)) - (start.tv_sec + ((double) start.tv_usec / 1000000.0)));
    } while (diff < duration);
}

