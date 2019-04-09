#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <chrono>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame, int runTime);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int stabilizingWindow = 6;
bool TLready, BRready, TLcleared, BRcleared;

Point TL, BR, screenPoint, centerGlobal;
vector<Point> centers;

/** @function main */
int main(int argc, const char** argv)
{
	VideoCapture capture(0);
	int w = capture.get(CAP_PROP_FRAME_WIDTH);
	int h = capture.get(CAP_PROP_FRAME_HEIGHT);
	Mat frame;

	TLready = false;
	BRready = false;
	TLcleared = false;
	BRcleared = false;

	auto start_time = chrono::high_resolution_clock::now();

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream
	if (capture.isOpened())
	{
		while (true)
		{
			capture >> frame;
			flip(frame, frame, 1);

			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{
				auto current_time = chrono::high_resolution_clock::now();
				int runTime = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
				detectAndDisplay(frame,runTime);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}

			int c = waitKey(1);
			if ((char)c == 'c') { break; }
		}
	}
	return 0;
}

Rect getLeftmostEye(vector<Rect> eyes)
{
	int leftmost = 99999999;
	int leftmostIndex = -1;
	for (int i = 0; i < eyes.size(); i++)
	{
		if (eyes[i].tl().x < leftmost)
		{
			leftmost = eyes[i].tl().x;
			leftmostIndex = i;
		}
	}
	return eyes[leftmostIndex];
}

int remap(int value, int start1, int stop1, int start2, int stop2)
{
	int outgoing = (start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1)));
	if (outgoing < 0)
		outgoing = -outgoing;
	return outgoing;
}

Vec3f getEyeball(Mat &eye, vector<Vec3f> &circles)
{
	vector<int> sums(circles.size(), 0);
	for (int y = 0; y < eye.rows; y++)
	{
		uchar *ptr = eye.ptr<uchar>(y);
		for (int x = 0; x < eye.cols; x++)
		{
			int value = static_cast<int>(*ptr);
			for (int i = 0; i < circles.size(); i++)
			{
				Point center((int)round(circles[i][0]), (int)round(circles[i][1]));
				int radius = (int)round(circles[i][2]);
				if (pow(x - center.x, 2) + pow(y - center.y, 2) < pow(radius, 2))
				{
					sums[i] += value;
				}
			}
			++ptr;
		}
	}
	int smallestSum = 9999999;
	int smallestSumIndex = -1;
	for (int i = 0; i < circles.size(); i++)
	{
		if (sums[i] < smallestSum)
		{
			smallestSum = sums[i];
			smallestSumIndex = i;
		}
	}
	return circles[smallestSumIndex];
}

Point stabilize(vector<Point> &points, int windowSize)
{
	float sumX = 0;
	float sumY = 0;
	int count = 0;
	for (int i = max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
	{
		sumX += points[i].x;
		sumY += points[i].y;
		++count;
	}
	if (count > 0)
	{
		sumX /= count;
		sumY /= count;
	}
	return Point(sumX, sumY);
}


/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, int runTime)
{
	vector<Rect> faces;
	vector<Rect> eyes;
	vector<Vec3f> circles;
	Vec3f eyeball;
	Mat eye;
	Mat frame_gray;
	Mat faceROI;

	centerGlobal.x = frame.cols / 2;
	centerGlobal.y = frame.rows / 2;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));

	if (faces.size() != 0)
	{
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame, faces[0].tl(), faces[0].br(), Scalar(255, 0, 255), 2);
		faceROI = frame_gray(faces[0]);
		eyes.clear();

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			//Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			//circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			rectangle(frame, faces[0].tl() + eyes[j].tl(), faces[0].tl() + eyes[j].br(), Scalar(255, 0, 0), 2);
		}
	}

	//Get leftmost eye
	if (eyes.size() > 0)
	{
		Rect leftEye = getLeftmostEye(eyes);
		eye = faceROI(leftEye); // crop the leftmost eye
		equalizeHist(eye, eye);
		HoughCircles(eye, circles, HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
		if (circles.size() > 0)
		{
			eyeball = getEyeball(eye, circles);
			Point center(eyeball[0], eyeball[1]);
			centers.push_back(center);
			if(centers.size()>stabilizingWindow)
				centers.erase(centers.begin());
			center = stabilize(centers, stabilizingWindow);
			centerGlobal = center;
			int radius = (int)eyeball[2];
			circle(frame, faces[0].tl() + leftEye.tl() + center, radius, cv::Scalar(0, 0, 255), 2);	
			circle(eye, center, radius, Scalar(255, 255, 255), 2);
		}
		imshow("Eye", eye);
	}

	if (runTime >= 4 && runTime < 7)
	{
		circle(frame, Point(15, 15), 15, Scalar(255, 255, 255), -1);
	}
	else if (runTime >= 7 && runTime < 12)
	{
		if(TLcleared == false)
		{ 
			centers.clear();
			TLcleared = true;
		}
		circle(frame, Point(15, 15), 15, Scalar(0, 255, 0), -1);
	}
	else if (runTime >= 12 && runTime < 15)
	{
		if (TLready == false)
		{
			TL = stabilize(centers, stabilizingWindow);
			TLready = true;
		}
		circle(frame, Point(frame.cols - 15, frame.rows - 15), 15, Scalar(255, 255, 255), -1);
	}
	else if (runTime >= 15 && runTime < 20)
	{
		if (BRcleared == false)
		{
			centers.clear();
			BRcleared = true;
		}
		circle(frame, Point(frame.cols - 15, frame.rows - 15), 15, Scalar(0, 255, 0), -1);
	}
	else if (runTime >= 20)
	{
		if (BRready == false)
		{
			BR = stabilize(centers, stabilizingWindow);
			BRready = true;
		}
		cout <<"BR X: "<< BR.x << "BR Y: " << BR.y << endl;
		cout << "TL X: " << TL.x << "TL Y: " << TL.y << endl;

		cout << "Global pos x: " << centerGlobal.x << "Global pos y: " << centerGlobal.y << endl;

		screenPoint.x = remap(centerGlobal.x, TL.x, BR.x, 15, frame.rows - 15);
		cout << "Screen point x: " << screenPoint.x<<endl;
		screenPoint.y = remap(centerGlobal.y, TL.y, BR.y, 15, frame.cols - 15);
		cout << "Screen point y: " << screenPoint.y << endl;
		circle(frame, screenPoint, 10, Scalar(0, 0, 255), -1);
	}

	imshow(window_name, frame);
}