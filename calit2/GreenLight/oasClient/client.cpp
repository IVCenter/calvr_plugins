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

oasclient::OASSound *heli, *boing;

int main(void)
{
    if (!oasclient::OASClientInterface::initialize("137.110.118.124", 31231))
    {
        std::cerr << "Could not set up connection to server!\n";
        exit(1);
    }

    //oasclient::OASClientInterface::sendFile("/home/schowkwa/OASClient", "LARGEFILE");
    // Set up bloop sound
    heli = new oasclient::OASSound("/home/schowkwa/OASClient/data", "helikopter.wav");

    if (!heli->isValid())
    {
        std::cerr << "Could not create helicopter sound!\n";
        exit(1);
    }

    //std::cerr << "Heli has handle " << heli->getHandle() << std::endl;

    test0();
    // Set up boing sound
    boing = new oasclient::OASSound("/home/schowkwa/OASClient/data", "BOING.WAV");

    if (!boing->isValid())
    {
        std::cerr << "Could not create boing sound!\n";
        exit(1);
    }

    std::cerr << "Boing has handle " << boing->getHandle() << std::endl;

    // BEGIN TEST 1 //
    // Loop in the x-y plane, starting from center, going to the right, and then counter-clockwise 360 degrees
    boing->play();
    wait(5);
	test1();


	// BEGIN TEST 2 //
	// Loop in the y-z plane, starting from directly in front, going overhead, then directly behind, then below, and back to the front.
	boing->play();
	wait(5);
    test2();


	// BEGIN TEST 3 //
    // Place sound in center, move directly forwards and then backwards, staying on the y axis
	boing->play();
	wait(5);
	test3();



	// BEGIN TEST 4 //
    // Doppler effect. Place sound forwards and to the left. Have sound move to the right, with a specified velocity.
	boing->play();
	wait(5);
    test4();



//    write(, "PTFI beach.WAV 258684", 256);
//    sendFile("/home/schowkwa/Desktop/samples/beach.WAV", );
    //wait(5);
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

	heli->setLoop(true);
	heli->play();
    wait(2);

    for (r = 0; r < 10; r += 0.25 / RESOLUTION)
    {
        heli->setPosition(r, 0, 0);
        wait(0.1 / RESOLUTION);
    }

    for (theta = 0; theta < 360; theta += 2.5 / RESOLUTION)
    {
        x = r * cos(theta * PI / 180);
        y = r * sin(theta * PI / 180);
        heli->setPosition(x, y, 0);
        wait(0.1 / RESOLUTION);
    }

    heli->stop();
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

/*
void sendFile(const char *filename, int socketFD)
{
    #define MAX_TRANSFER_SIZE 512
    
    int fileSize;
    struct stat fileInfo;
    char *data;
    
    if (stat(filename, &fileInfo) != 0) // stat returns 0 on success
    {
        std::cerr << "Could not find file \"" << filename << "\"!\n";
        return;
    }
    
    fileSize = fileInfo.st_size;
    
    // Read file from disk
    data = new char[fileSize];
    std::ifstream fileIn(filename, std::ios::in | std::ios::binary);
    fileIn.read(data, fileInfo.st_size);
    
    ///////////////////////////
    // Send file over socket //
    ///////////////////////////
    
    int bytesLeft, bytesWritten, bytesToWrite = MAX_TRANSFER_SIZE;
    char *dataPtr;
    bytesLeft = fileSize;
    dataPtr = data;
    
    std::cerr << "********** Sending file \"" << filename << "\" over socket **********\n";

    long count = 0;

    while (bytesLeft > 0)
    {
        if (bytesLeft < MAX_TRANSFER_SIZE)
        {
            bytesToWrite = bytesLeft;
        }
    
        bytesWritten = write(socketFD, dataPtr, bytesToWrite);
        
        if (bytesWritten == 0 || bytesWritten == -1)
        {
            std::cerr << "Error occurred sending file to server!\n";
            break;
        }
        dataPtr += bytesWritten;
        bytesLeft -= bytesWritten;
        
        if (count++ % 20 == 0)    
        {
            float percentage = ((float) (fileSize - bytesLeft) / fileSize) * 100.0;
            std::cerr << "---> File Transfer " << percentage << "% complete\n";
        }
    }
        
    std::cerr << "********** File Transmission Complete **********\n";


}
*/
