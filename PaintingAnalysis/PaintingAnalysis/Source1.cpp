/*
Skye McKay
Final Project
Painting Analysis
This is the program that I'll use to extract the data from the images in order to make a matrix on vectors from that information.
With that matrix I intend to create lagrange multipliers alpha and constants b by running them through a SVM program I made in a Machine Learning class.
With these weights I should be able to write an accuarate clustering code which will be a different program all together.
*/

#include "opencv2/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>

using namespace cv;
using namespace std;

double lowestDis(vector<Point2i>cent, Mat &img);

int detectAndDisplay(Mat frame,int ppl);
String face_cascade_name = "haarcascade_frontalface_alt_tree.xml";
String profile_cascade_name = "haarcascade_profileface.xml";
String eyes_cascade_name = "frontalEyes35x16.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier profile_cascade;
String window_name = "Capture - Face detection";

int myMin(int a, int b, int c);
int myMax(int a, int b, int c);
void mySkinDetect(Mat& src, Mat& dst,Mat& pink,int bri);
int centerPeople(vector<vector<Point2i>> &blobs, vector<Point2i> &cent, int avg, int adder);

double Pink(Mat &img, Mat &dest,int bri);
double Blue(Mat &img, Mat &dest,int bri);
double Green(Mat &img, Mat &dest);

int findObjects(const Mat &binary, vector<vector<Point2i>> &blobs);

double Saturation(Mat &img);
double Brightness(Mat &img);

double Nudity(Mat &img, int ppl);

int main(int argc, char **argv){

	Mat img = imread("Titians work/painting90.jpg");
	if (!img.data) {
		cout << "File not found" << std::endl;
		return -1;
	}
	Mat dest = Mat::zeros(img.size(), CV_8UC1);
	Mat pink = Mat::zeros(img.size(), CV_8UC1);

	
	double B = Brightness(img);
	double S = Saturation(img);
	cout << "The average Brightness of this painting is " << B << endl;
	cout << "The average Saturation of this painting is " << S << endl;
	
	if (S >= 2){

		if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
		if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
		if (!profile_cascade.load(profile_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };

		double P = Pink(img, pink, B);
		int erosion_size = 2;
		int dilation_size = 2;
		Mat element = getStructuringElement(MORPH_RECT,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));


		mySkinDetect(img, dest, pink, B);

		cout << "Amount of pink is " << P*10 << endl;
		vector<Point2i> cent;
		vector <vector<Point2i>> blobs;
		Mat binary;
		threshold(dest, binary, 0.0, 1.0, THRESH_BINARY);
		dilate(binary, binary, element);
		erode(binary, binary, element);

		int avg = findObjects(binary, blobs);
		int adder = 0;
		if (avg > 300){
			adder = 100;
		}
		if (avg < 200){
			adder = -75;
		}

		int ppl = centerPeople(blobs, cent, avg, adder);

		if (ppl == 0){
			ppl = 1;
		}

		Mat label = Mat::zeros(dest.size(), CV_8UC1);


		for (size_t i = 0; i < blobs.size(); i++) {
			for (size_t j = 0; j < blobs[i].size(); j++) {
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;

				label.at<uchar>(y, x) = 255;
			}
		}

		cout << "number of people is " << ppl << endl;

		double nude = Nudity(dest, ppl);
		cout << "nudity percentage is " << nude << endl;

		Mat blue = Mat::zeros(img.size(), CV_8UC1);

		Mat green = Mat::zeros(img.size(), CV_8UC1);
		double g = Green(img, green);
		double bl = Blue(img, blue, B);
		double dis = lowestDis(cent, img);
		cout << "Amount of blue is " << bl*10 << endl;
		cout << "The lowest distance from the center is " << dis << endl;
		cout << "Amount of green is " << g << endl;



		namedWindow("Original");
		namedWindow("binary");
		namedWindow("pink");
		namedWindow("blue");

		imwrite("Blue2.jpg", blue);
		imwrite("green2.jpg", green);
		imwrite("skin2.jpg", dest);
		imwrite("Pink2.jpg", pink);


		imshow("Original", img);
		imshow("binary", dest);
		imshow("pink", pink);
		imshow("blue", blue);

		int face = detectAndDisplay(img,ppl);
		imwrite("Face2.jpg", img);

		cout << "Number of readable faces is " << face << endl;

		waitKey(0);
	}
	else {
		namedWindow("Original");
		imshow("Original", img);
		cout << "Set numbers for the sketches "<< endl;

		waitKey(0);
	}
	return 0;

}

int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst, Mat &pink,int bri) {
	int average;
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			//For each pixel, compute the average intensity of the 3 color channels
			Vec3b intensity = src.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			average = (B + G + R) / 3;
			if (R> B && R > G && R>100 && G > 78 && B > 68  &&  abs(R-B)> 32 && B < 150 && R > average && average > 109 && pink.at<uchar>(i,j)== 0 &&((bri <40&& B+G<250)||(bri>=40&&bri<50)||(bri>=50 &&abs(R-average)>38))){
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

int findObjects(const Mat &binary, vector<vector<Point2i>> &blobs){
	blobs.clear();
	int num=0;
	Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int labelCount = 2;

	for (int y = 0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 0; x < label_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}
			
			Rect rect;
			floodFill(label_image, Point(x, y), labelCount, &rect, 0, 0, 4);

			vector <Point2i> blob;
			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != labelCount) {
						continue;
					}
					blob.push_back(Point2i(j, i));
				}
			}
			if (blob.size()>80){
				blobs.push_back(blob);
				num += static_cast<int>(blob.size());
				labelCount++;
				
			}

		}
	}
	int size = static_cast<int>(blobs.size());
	if (size == 0){
		return 0;
	}
	return num / size;
}


double Brightness(Mat &img){
	Mat frame_gray;
	cvtColor(img, frame_gray, COLOR_BGR2GRAY);
	double result = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			result += img.at<uchar>(i, j);
		}
	}
	return result / (img.rows * img.cols);
}

double Green(Mat &img,Mat &dest){
	double green = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			Vec3b intensity = img.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if (myMax(R, G, B) == G &&  G > 40 && B +R< 250){
				dest.at<uchar>(i, j) = 255;
				green++;
			}
		}
	}
	return green / (img.rows*img.cols);
}

double Saturation(Mat &img){
	double average, result, max;
	result = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			Vec3b intensity = img.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			average = (B + G + R) / 3;
			max = myMax(R, G, B);
			result += max - average;
		}
	}
	return result / (img.rows * img.cols);
}

double Nudity(Mat &img, int ppl){
	double num = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols;j++){
			if (img.at<uchar>(i, j) == 255){
				num += 1;
			}
		}
	}
	
	return num / (ppl *10000) ;
}

double Pink(Mat &img,Mat &dest,int bri){
	double total = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			Vec3b intensity = img.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if (R> B && R > G && R > 100 && G < 58){
				dest.at<uchar>(i, j) = 255;
				total++;
			}
		}
	}
	return total /(img.rows*img.cols);
}

double Blue(Mat &img, Mat &dest,int bri){
	double total = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			Vec3b intensity = img.at<Vec3b>(i, j); //Vec3b is a vector of 3 uchar (unsigned character)
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			if (B> R && B > G && ((bri < 50 && B < 90 && B>25 && R < 38 && abs(B-G)>4)|| B>90 &&bri>50) ){
				dest.at<uchar>(i, j) = 255;
				total++;
			}
		}
	}
	return total / (img.rows*img.cols);
}

int centerPeople(vector<vector<Point2i>> &blobs, vector<Point2i> &cent, int avg,int adder){
	cent.clear();
	int ppl = 0;
	for (size_t i = 0; i < blobs.size(); i++){
		if (static_cast<int>(blobs[i].size())>avg + adder){
			ppl += 1;
			int x_c = 0;
			int y_c = 0;
			int size = 0;
			for (size_t j = 0; j < blobs[i].size(); j++){
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;
				x_c += x;
				y_c += y;
				size++;
			}
			cent.push_back(Point2i(x_c / size, y_c / size));
		}
	}
	return ppl;
}

double lowestDis(vector<Point2i>cent, Mat &img){
	int centerx = img.rows / 2;
	int centery = img.cols / 2;
	double dis = 1000000000;
	double temp = 0;
	for (size_t i = 0; i < cent.size(); i++){
		int curx = cent[i].x;
		int cury = cent[i].y;
		temp = sqrt(pow((centerx - curx), 2) + pow((centery - cury), 2));
		if (temp < dis){
			dis = temp;
		}
	}
	if (temp == 0){
		return 0;
	}
	else {
		return dis;
	}
}

int detectAndDisplay(Mat frame,int ppl)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	int amount = 0;
	for (size_t i = 0; i < faces.size(); i++)
	{
		amount++;
		ppl--;
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;
		//-- In each face, detect eyes
		
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	/*
	if (ppl < 1){
		profile_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			amount++;
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
			Mat faceROI = frame_gray(faces[i]);
			std::vector<Rect> eyes;
			//-- In each face, detect eyes
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			}
		}
	}
	*/
	//-- Show what you got
	imshow(window_name, frame);
	return amount;
}

